"""Shape and plumbing self test on a randomly initialised model.

No weights download and no GPU needed. Random weights do not refuse, so this
says nothing about whether the method works; it says the maths, the hooks, the
edit and the metrics are wired correctly, which is what you want settled before
a run spends GPU hours on it.

  python scripts/selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from ablate import calibration as CAL
from ablate import subspace as S
from ablate.apply import (ConditionalAblation, EditPlan, apply_plan,
                          condition_pooled, fit_condition_probe, naive_plan,
                          probe_scores, restore, snapshot, tau_for_fpr)
from ablate.quant import (PRESETS, edit_survival, quantize_dequantize,
                          quantize_model_, restore_linears, snapshot_linears)
from ablate.capture import capture_residuals, generate, student_kl, teacher_trace
from ablate.evalharness import EvalSets, calibration_regression, evaluate, objective
from ablate.modelio import decoder_layers, format_chat, n_layers, residual_writers

DEV = "cpu"
OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + detail) if detail else ''}")


def tiny_model():
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, intermediate_size=256,
                      num_hidden_layers=6, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).to(DEV).eval()
    m.requires_grad_(False)
    return m, tok


def main():
    print("building a tiny random Qwen2")
    model, tok = tiny_model()
    L, H = n_layers(model), model.config.hidden_size
    print(f"  layers {L}  hidden {H}  vocab {model.config.vocab_size}\n")

    print("modelio")
    ws = residual_writers(model)
    check("residual writers resolved", len(ws) == 2 * L, f"{len(ws)} for {L} layers")
    check("writer output dim is hidden",
          all(w.module.weight.shape[0] == H for w in ws))
    check("embedding not among writers",
          all("embed" not in w.name for w in ws))
    check("chat template renders", "<|im_start|>" in format_chat(tok, "hello"))

    print("\ncapture")
    prompts = ["What is the capital of France?", "Explain gravity briefly.",
               "List three fruits.", "Why is the sky blue?"]
    acts = capture_residuals(model, tok, prompts, batch_size=2, device=DEV)
    check("residual shape", tuple(acts.shape) == (4, L + 1, H), str(tuple(acts.shape)))
    check("residuals finite", bool(torch.isfinite(acts).all()))

    a2 = capture_residuals(model, tok, prompts, batch_size=4, device=DEV)
    check("capture invariant to batch size",
          torch.allclose(acts, a2, atol=2e-3), f"max diff {float((acts-a2).abs().max()):.2e}")

    print("\nsubspace")
    torch.manual_seed(1)
    pos = torch.randn(40, L + 1, H)
    direction = torch.randn(H)
    direction = direction / direction.norm()
    neg = pos - 3.0 * direction                      # a planted rank 1 difference
    basis, var, sep = S.difference_subspace(pos, neg, k=2, paired=True)
    cos = float((basis[3, 0] @ direction).abs())
    check("recovers a planted direction", cos > 0.99, f"cos {cos:.4f}")
    check("leading component dominates", float(var[3, 0]) > 0.9, f"{float(var[3,0]):.3f}")
    check("rows orthonormal",
          torch.allclose(basis[3] @ basis[3].T, torch.eye(2), atol=1e-4))
    check("separation positive", float(sep[3]) > 1.0, f"d {float(sep[3]):.2f}")

    protected = {"fake": (direction / direction.norm()).expand(L + 1, H).contiguous()}
    ss_p = S.build(pos, neg, k=2, protected=protected, protect=True)
    ss_n = S.build(pos, neg, k=2, protected=protected, protect=False)
    live = ss_p.layer(3)
    leak = float((live @ direction).abs().max()) if live.shape[0] else 0.0
    check("protection removes the protected direction", leak < 1e-4, f"leak {leak:.2e}")
    check("protection drops rank here", int(ss_p.rank[3]) < int(ss_n.rank[3]),
          f"{int(ss_p.rank[3])} vs {int(ss_n.rank[3])}")
    check("overlap diagnostic recorded",
          "fake" in ss_p.protected_overlap
          and abs(float(ss_p.protected_overlap["fake"][3, 0])) > 0.9)
    check("with_rank clamps", ss_n.with_rank(1).layer(3).shape[0] == 1)
    check("best_layer inside window", 0 <= ss_p.best_layer() < L + 1)

    print("\napply")
    real = S.build(acts, acts.flip(0), k=1, protect=False)
    orig = snapshot(model)
    plan = EditPlan(center=3.0, width=0.7, peak=1.0, kernel="gaussian",
                    norm_restore="column")
    alphas = [round(plan.alpha(i), 3) for i in range(L)]
    check("gaussian kernel peaks at centre", alphas[3] == max(alphas), str(alphas))
    check("kernel decays off centre", alphas[0] < 0.05)

    w0 = ws[6].module.weight.detach().clone()
    rec = apply_plan(model, real, plan, originals=orig)
    check("narrow kernel edits strictly fewer matrices than a flat plan",
          0 < rec["n_edited"] < 2 * L, f"{rec['n_edited']} of {2*L}")
    changed = not torch.equal(w0, ws[6].module.weight)
    check("weights actually changed", changed)

    # pick the writer at the kernel centre, where alpha is 1.0; a writer at the
    # edge is barely touched and would make any residual assertion vacuous
    hottest = max(rec["edited"], key=lambda e: e["alpha"])
    assert hottest["alpha"] > 0.99, hottest
    w = next(x for x in residual_writers(model) if x.name == hottest["name"])
    W0 = orig[w.name].float()
    Q = real.layer(w.layer).float()

    def resid_of(M):
        return float((Q @ M).norm() / (Q @ W0).norm().clamp(min=1e-12))

    col = w.module.weight.detach().float().clone()   # clone: copy_ would alias it
    check("column norms preserved",
          torch.allclose(W0.norm(dim=0), col.norm(dim=0), rtol=2e-2),
          f"max rel {float(((col.norm(dim=0)-W0.norm(dim=0)).abs()/W0.norm(dim=0).clamp(min=1e-9)).max()):.2e}")
    check("column restore keeps the subspace ablated",
          resid_of(col) < 1e-5, f"residual {resid_of(col):.2e}")

    apply_plan(model, real, EditPlan(center=3.0, width=0.7, peak=1.0,
                                     norm_restore="none"), originals=orig)
    none_r = resid_of(w.module.weight.detach().float())
    apply_plan(model, real, EditPlan(center=3.0, width=0.7, peak=1.0,
                                     norm_restore="row"), originals=orig)
    row_m = w.module.weight.detach().float().clone()
    row_r = resid_of(row_m)
    check("row restore preserves row norms",
          torch.allclose(W0.norm(dim=1), row_m.norm(dim=1), rtol=2e-2))
    check("row restore leaks the ablated subspace back, column restore does not",
          row_r > 10 * max(resid_of(col), none_r),
          f"row {row_r:.2e} vs column {resid_of(col):.2e}")
    check("both restores return the Frobenius norm",
          abs(float(row_m.norm()/W0.norm()) - 1) < 0.05
          and abs(float(col.norm()/W0.norm()) - 1) < 0.05)

    restore(model, orig)
    check("restore is exact",
          all(torch.equal(orig[w.name], w.module.weight.detach().cpu())
              for w in residual_writers(model)))

    nv = naive_plan(model, real)
    check("naive plan is flat, full strength, embedding included",
          nv.kernel == "uniform" and nv.peak == 1.0 and nv.edit_embedding
          and nv.norm_restore == "none")

    print("\nconditional hook, per sequence gate")
    restore(model, orig)
    base_out = generate(model, tok, prompts[:2], max_new_tokens=4, device=DEV)
    hooks_before = [len(l._forward_hooks) for l in decoder_layers(model)]
    cl = 3
    pooled = condition_pooled(model, tok, prompts, cl, n_suffix=2, batch_size=2, device=DEV)
    check("pooled instruction activations", tuple(pooled.shape) == (4, H), str(tuple(pooled.shape)))
    probe = fit_condition_probe(pooled[:2], pooled[2:])
    check("gate probe returns weight, bias and auc",
          probe["weight"].shape == (H,) and isinstance(probe["bias"], float))
    sc = probe_scores(pooled, probe)
    check("probe separates the two groups it was fit on",
          float(sc[:2].mean()) > float(sc[2:].mean()))

    big = EditPlan(center=4.0, width=1.0, peak=1.0)
    with ConditionalAblation(model, real, big, probe["weight"], probe["bias"],
                             tau=1e9, condition_layer=cl, n_suffix=2) as ca:
        hi = generate(model, tok, prompts[:2], max_new_tokens=4, device=DEV)
    check("unreachable tau is an exact no-op", hi == base_out)
    check("gate never fired", ca.n_fired == 0, f"{ca.n_fired}/{ca.n_seq}")
    check("gate counts whole sequences, not tokens", ca.n_seq == 2, str(ca.n_seq))

    # assert on logits, not decoded text: a rank-1 projection out of 128 dims
    # perturbs the distribution without necessarily moving a greedy argmax over
    # a 151k vocabulary within four tokens
    enc = tok([format_chat(tok, p) for p in prompts[:2]], return_tensors="pt",
              padding=True, add_special_tokens=False)
    with torch.no_grad():
        clean = model(**enc).logits.clone()
    with ConditionalAblation(model, real, big, probe["weight"], probe["bias"],
                             tau=-1e9, condition_layer=cl, n_suffix=2) as ca2:
        with torch.no_grad():
            gated = model(**enc).logits.clone()
    check("gate fires on every sequence below tau", ca2.trigger_rate == 1.0,
          f"{ca2.trigger_rate:.2f}")
    delta = float((gated - clean).abs().max())
    check("always-on gate perturbs the logits", delta > 1e-4, f"max logit delta {delta:.3e}")
    check("hooks removed on exit",
          [len(l._forward_hooks) for l in decoder_layers(model)] == hooks_before)

    hs = probe_scores(condition_pooled(model, tok, prompts[2:], cl, n_suffix=2,
                                       batch_size=2, device=DEV), probe)
    tau = tau_for_fpr(hs, 0.5)
    check("tau_for_fpr sits inside the score range",
          float(hs.min()) <= tau <= float(hs.max()), f"tau {tau:.3f}")

    print("\nquantization")
    W = torch.randn(64, 256) / 16
    for name, kw in PRESETS.items():
        Q = quantize_dequantize(W, **kw)
        if kw["bits"] >= 16:
            check(f"{name} is a no-op", torch.equal(Q, W))
            continue
        gs = kw["group_size"] or W.shape[-1]
        levels = len(torch.unique(Q[0, :gs]))
        check(f"{name} uses at most 2^bits levels per group",
              levels <= 2 ** kw["bits"], f"{levels} levels")
    err4 = float((quantize_dequantize(W, 4, 128) - W).pow(2).sum().sqrt())
    err8 = float((quantize_dequantize(W, 8, 128) - W).pow(2).sum().sqrt())
    check("more bits is less error", err8 < err4, f"int8 {err8:.4f} < int4 {err4:.4f}")

    lin = snapshot_linears(model)
    st = quantize_model_(model, bits=4, group_size=128)
    check("quantize_model_ touched the decoder linears", st["tensors"] > 0, str(st["tensors"]))
    check("quantization error is nonzero and sane",
          0 < st["rel_weight_rmse"] < 0.5, f"{st['rel_weight_rmse']:.4f}")
    n = restore_linears(model, lin)
    check("restore_linears is exact",
          n > 0 and all(torch.equal(lin[k], dict(model.named_modules())[k].weight.detach().cpu())
                        for k in list(lin)[:5]))

    W0 = torch.randn(64, 256) / 16
    surv = edit_survival(W0, W0, bits=4, group_size=128)
    check("a zero edit reports zero norm", surv["delta_norm"] == 0.0)
    W1 = W0 + 0.02 * torch.randn_like(W0)
    s4 = edit_survival(W0, W1, bits=4, group_size=128)
    s8 = edit_survival(W0, W1, bits=8, group_size=128)
    check("edit survives in the mean at both precisions",
          0.5 < s4["retained"] < 1.5 and 0.8 < s8["retained"] < 1.2,
          f"int4 {s4['retained']:.3f} int8 {s8['retained']:.3f}")
    check("coarser grid adds more orthogonal noise", s4["ratio"] > s8["ratio"],
          f"int4 ratio {s4['ratio']:.2f} > int8 {s8['ratio']:.2f}")

    print("\nKL")
    trace = teacher_trace(model, tok, prompts, max_new_tokens=6, top_k=64,
                          batch_size=2, device=DEV)
    check("trace shapes line up",
          trace["topk_logprobs"].shape[:2] == (4, 6)
          and trace["full_ids"].shape[0] == 4)
    k0 = student_kl(model, trace, batch_size=2, device=DEV)
    check("KL against itself is ~0", k0["kl_mean"] < 1e-4, f"{k0['kl_mean']:.3e}")
    apply_plan(model, real, EditPlan(center=3.0, width=6.0, peak=1.0,
                                     kernel="uniform", norm_restore="none"),
               originals=orig)
    k1 = student_kl(model, trace, batch_size=2, device=DEV)
    check("KL rises after a heavy edit", k1["kl_mean"] > k0["kl_mean"],
          f"{k0['kl_mean']:.3e} -> {k1['kl_mean']:.3e}")
    restore(model, orig)

    print("\ncalibration and objective")
    items = CAL.items()
    check("96 calibration items in four categories",
          len(items) == 96 and len({i["category"] for i in items}) == 4)
    check("constraint checker catches a violation",
          CAL.check_constraint("the water cycle moves water", {"type": "absent", "value": "water"}) is False)
    check("constraint checker passes a clean response",
          CAL.check_constraint("rain falls and evaporates", {"type": "absent", "value": "water"}) is True)
    check("no_letter check works",
          CAL.check_constraint("brains", {"type": "no_letter", "value": "e"}) is True
          and CAL.check_constraint("beans", {"type": "no_letter", "value": "e"}) is False)

    base = {"refusal_rate": 0.9, "over_refusal_rate": 0.2, "kl_mean": 0.0,
            "calibration": {"unanswerable": {"behaviour_rate": 0.8},
                            "nonsense": {"behaviour_rate": 0.7}}}
    good = {"refusal_rate": 0.05, "over_refusal_rate": 0.2, "kl_mean": 0.01,
            "calibration": {"unanswerable": {"behaviour_rate": 0.78},
                            "nonsense": {"behaviour_rate": 0.70}}}
    bad = {"refusal_rate": 0.05, "over_refusal_rate": 0.2, "kl_mean": 0.01,
           "calibration": {"unanswerable": {"behaviour_rate": 0.10},
                           "nonsense": {"behaviour_rate": 0.05}}}
    og, ob = objective(base, good), objective(base, bad)
    check("objective punishes calibration collapse at equal refusal and KL",
          og["value"] < ob["value"], f"{og['value']:.3f} vs {ob['value']:.3f}")
    reg = calibration_regression(base, bad)
    check("regression names the worst category",
          reg["worst_category"] in ("unanswerable", "nonsense")
          and reg["worst_drop"] > 0.6)
    check("improvements do not count as damage",
          calibration_regression(base, {"calibration": {
              "unanswerable": {"behaviour_rate": 1.0},
              "nonsense": {"behaviour_rate": 1.0}}})["mean_drop"] == 0.0)

    print("\nend to end evaluate")
    sets = EvalSets(harmful=prompts[:2], harmless=prompts[2:], borderline=prompts[:1],
                    calibration=items[:4])
    tr = teacher_trace(model, tok, sets.harmless, max_new_tokens=4, top_k=32,
                       batch_size=2, device=DEV)
    res = evaluate(model, tok, sets, trace=tr, batch_size=2, device=DEV, max_new_tokens=8)
    check("evaluate returns every metric",
          all(k in res for k in ("refusal_rate", "over_refusal_rate",
                                 "kl_mean", "calibration")), str(sorted(res)))

    print(f"\n{len(OK)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
