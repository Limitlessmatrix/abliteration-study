"""Abliterate one model end to end and write the artifact.

Stages 2 and 4 pick the edit, Stage 5 heals it, Stage 6 scores and packages it.
Every stage writes to the run directory and checks for its predecessor's output
first, so an interrupted run resumes instead of starting over.

  python scripts/abliterate.py --model <id or dir::file.gguf> --out runs/<name>
  python scripts/abliterate.py --config configs/v32_kljgrid.json --model Qwen/Qwen3-8B --out runs/qwen3-8b-v32

--config FILE.json sets argument defaults (keys are the argument names with underscores, e.g. "refusal_tol");
flags on the command line still win. configs/ holds the settings of every version in the study (v2, v3 kl_first,
v3 pareto, v3.1 kljudge, v3.2 kljgrid). --study-journal auto puts the resumable Optuna journal in the run directory.
--guard-judge accepts a local directory or a Hub id with a pinned revision ("Qwen/Qwen3Guard-Gen-8B@<sha>").

The only part that needs a decision rather than a measurement is whether to
ship baked weights or the conditional hook, which --artifact selects.
"""
from __future__ import annotations

import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from ablate import datasets as D
from ablate import layerscan
from ablate import subspace as S
from ablate.apply import (EditPlan, apply_plan, condition_pooled,
                          fit_condition_probe, naive_plan, probe_scores,
                          restore, snapshot, tau_for_fpr)
from ablate.capture import teacher_trace
from ablate.evalharness import evaluate, objective
from ablate.hookedit import HookEdit
from ablate.heal import (attach_lora, attach_qat, heal, merge_and_save,
                         write_model_card)
from ablate.modelio import hidden_size, load, n_layers, n_suffix_tokens, residual_writers
from ablate.pipeline import build_eval_sets, load_subspace, save_subspace
from ablate.quant import (SKIP, _targets, PRESETS, quantize_model_, restore_linears,
                          snapshot_linears, survival_report)
from ablate.refusal import OllamaJudge
from ablate.search import best_plan, make_storage, pareto, rank1_grid, run_study

from ablate.config import data_dir, resolve_model_dir
DATA = data_dir()   # ABLATE_DATA or --data, default <repo>/data


def log(m):
    print(m, flush=True)


def stage_done(out: Path, name: str) -> bool:
    return (out / f"{name}.done").exists()


def mark(out: Path, name: str, payload: dict):
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
    (out / f"{name}.done").touch()


def ship_and_reload(model, tok, wdir: Path, a, ev) -> dict:
    """Write the checkpoint a user would download, then score that file.

    The in memory numbers measure a model this process built. The shipped
    artifact goes through save_pretrained, a fresh from_pretrained and whatever
    the config round trip does to tied embeddings, nested text_config and dtype.
    Loading it back cold and rerunning the harness is the only way to know the
    two are the same model. ev reads the module level model through
    _EVAL_MODEL, so the reloaded copy is scored with the same sets and trace.
    """
    wdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(wdir, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(wdir)
    gb = sum(f.stat().st_size for f in wdir.glob("*.safetensors")) / 1e9
    log(f"checkpoint -> {wdir}  {gb:.2f} GB")
    # park the in memory copy in host RAM so only one model is on the GPU;
    # two bf16 copies do not fit for a 27B on 80 GB or a 4B on 24 GB
    park = a.device != "cpu"
    if park:
        model.to("cpu"); torch.cuda.empty_cache()
    fresh, _ = load(str(wdir), dtype=a.dtype, device=a.device)
    worst = 0.0
    ref = dict(model.named_parameters())
    for n, p in fresh.named_parameters():
        q = ref.get(n)
        if q is not None and q.shape == p.shape:
            worst = max(worst, float((p.detach().float() - q.detach().float().to(p.device)).abs().max()))
    log(f"  reloaded, max abs weight diff vs in memory {worst:.3e}")
    _EVAL_MODEL[0] = fresh
    try:
        r = ev()
    finally:
        _EVAL_MODEL[0] = None
        del fresh
        torch.cuda.empty_cache()
        if park:
            model.to(a.device)
    r["_ckpt"] = {"dir": str(wdir), "gb": round(gb, 3), "max_abs_weight_diff": worst}
    log(f"  ckpt refusal {r.get('refusal_rate'):.3f}  KL {r.get('kl_mean', float('nan')):.4f}")
    return r


_EVAL_MODEL = [None]


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="JSON file of argument defaults (see configs/)")
    ap.add_argument("--data", default="", help="directory with the built search sets; default $ABLATE_DATA or <repo>/data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--cal", type=int, default=96)
    ap.add_argument("--k", type=int, default=1,
                    help="subspace rank; 1 is difference of means, which is what works")
    ap.add_argument("--contrast", default="bulk",
                    choices=["bulk", "matched", "refuse_vs_comply"])
    ap.add_argument("--clip", type=float, default=1.0,
                    help="Winsorise activations at this percentile; 0.995 for Gemma")
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--kl-tokens", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--kl-budget", type=float, default=0.25)
    ap.add_argument("--objective", default="scalar", choices=["scalar", "pareto", "kl_first"],
                    help="pareto = v3 two objective search with a KL target (see ablate.search)")
    ap.add_argument("--kl-target", type=float, default=0.01)
    ap.add_argument("--refusal-tol", type=float, default=0.02,
                    help="pareto pick: lowest KL trial with refusal rate at or under this")
    ap.add_argument("--cal-tol", type=float, default=0.05)
    ap.add_argument("--startup-trials", type=int, default=0, help="0 = 30 percent of --trials")
    ap.add_argument("--study-journal", default="",
                    help="Optuna journal file ('auto' = <out>/study.journal); an interrupted run resumes its trials from it")
    ap.add_argument("--quant", default="int4-g128",
                    help="deployment precision; search and layer scan use it")
    ap.add_argument("--final-quant", default="int8,int4-g128")
    ap.add_argument("--heal-steps", type=int, default=0, help="0 disables Stage 5")
    ap.add_argument("--heal-lr", type=float, default=1e-4)
    ap.add_argument("--heal-corpus", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--heal-reproject", action=argparse.BooleanOptionalAction, default=True,
                    help="project the merged writers against the subspace again after healing")
    ap.add_argument("--heal-probes", action="store_true",
                    help="put calibration probes in the heal corpus; off because the "
                         "untouched teacher refuses some of them")
    ap.add_argument("--keep-generations", action="store_true", default=True,
                    help="store every generation so a judge can rescore later")
    ap.add_argument("--qat", action="store_true", help="heal through the quant grid")
    ap.add_argument("--artifact", default="baked", choices=["baked", "hook", "none"])
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--judge", default="")
    ap.add_argument("--guard-judge", default="",
                    help="Qwen3Guard-Gen: local dir or Hub id[@revision]; refusal in every evaluation is scored by it (ablate.guardjudge)")
    ap.add_argument("--guard-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="TPE sampler seed of the search (repeat runs for variance)")
    ap.add_argument("--grid", default="", choices=["", "rank1"],
                    help="rank1 = start the search with the fixed 24 cell rank 1 band grid (ablate.search.rank1_grid)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--fpr", type=float, default=0.02)
    ap.add_argument("--edit-mode", default="weights", choices=["weights", "hook"],
                    help="hook: search and score through forward hooks, no weight is written or "
                         "snapshotted. For models where the snapshot or the model itself does not "
                         "fit; bake afterwards with scripts/bake_stream.py")
    ap.add_argument("--bnb", default="", choices=["", "4bit", "8bit"],
                    help="hold the model bitsandbytes quantized; needs --edit-mode hook")
    if known.config:
        cfg = json.loads(Path(known.config).read_text())
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}   # "_comment" style keys are notes
        dests = {x.dest for x in ap._actions}
        unknown = sorted(set(cfg) - dests)
        if unknown:
            sys.exit(f"--config {known.config}: unknown keys {unknown}")
        ap.set_defaults(**cfg)
    a = ap.parse_args()
    global DATA
    if a.data:
        DATA = Path(a.data)
    if a.study_journal == "auto":
        a.study_journal = str(Path(a.out) / "study.journal")
    hook = a.edit_mode == "hook"
    if a.bnb and not hook:
        sys.exit("--bnb needs --edit-mode hook: quantized weights cannot be rewritten")
    if hook and (a.quant or a.final_quant or a.heal_steps or a.save_weights):
        sys.exit("--edit-mode hook needs --quant '' --final-quant '' and no --heal-steps or "
                 "--save-weights; materialize weights with scripts/bake_stream.py")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"=== {a.model} -> {out} ===")
    model, tok = load(a.model, dtype=a.dtype, device=a.device, bnb=a.bnb)
    L, n_suf = n_layers(model), n_suffix_tokens(tok)
    log(f"{L} layers, hidden {hidden_size(model)}, {n_suf} suffix tokens")

    judge = None
    if a.judge:
        judge = OllamaJudge(a.judge)
        if not judge.available():
            log(f"judge {a.judge} unavailable, markers only"); judge = None
    if a.guard_judge:
        from ablate.guardjudge import GuardJudge
        a.guard_judge = resolve_model_dir(a.guard_judge)
        judge = GuardJudge(a.guard_judge, device="cuda" if a.device == "auto" else a.device, batch_size=a.guard_batch)
        log(f"refusal judged by Qwen3Guard at {a.guard_judge}")

    # ---------- Stage 2: subspace ----------
    if stage_done(out, "extract"):
        log("stage 2 cached")
        variants = {n: load_subspace(out / f"subspace_{n}.pt")
                    for n in ("protected", "plain") if (out / f"subspace_{n}.pt").exists()}
        meta = json.loads((out / "extract.json").read_text())
    else:
        from ablate.capture import capture_residuals
        from ablate import calibration as CAL
        tr = D.load(DATA, split="train")
        pairs, cal_all = tr["pairs"][: a.pairs], tr["calibration"]
        border = tr["borderline"][: a.pairs]
        log(f"labelling {len(pairs)} pairs, {len(border)} borderline")
        D.label_behaviour(model, tok, pairs, "harmful", device=a.device, batch_size=a.batch)
        D.label_behaviour(model, tok, pairs, "harmless", device=a.device, batch_size=a.batch)
        D.label_behaviour(model, tok, border, "prompt", device=a.device, batch_size=a.batch)
        cap = lambda ps: capture_residuals(model, tok, ps, batch_size=a.batch,
                                           device=a.device, positions="post_instruction",
                                           clip=a.clip)
        A = {"harmful": cap([p["harmful"] for p in pairs]),
             "harmless": cap([p["harmless"] for p in pairs]),
             "cal_probe": cap([c["probe"] for c in cal_all]),
             "cal_control": cap([c["control"] for c in cal_all]),
             "borderline": cap([b["prompt"] for b in border])}
        protected = {}
        for c in CAL.CATEGORIES:
            idx = [i for i, x in enumerate(cal_all) if x["category"] == c]
            if len(idx) >= 4:
                protected[c] = S.behaviour_directions(A["cal_probe"][idx], A["cal_control"][idx])
        if a.contrast == "bulk":
            # Unmatched bulk contrast: harmful behaviours against an unrelated
            # harmless corpus. This is what Arditi, FailSpy, Heretic and
            # Grimjim all use, and it is the configuration that works.
            #
            # Our matched twins were the opposite bet, and the bet lost.
            # Petrov (arXiv:2603.22061) tested topic matched harmless prompts on
            # Qwen3.5-2B and found directions that "fail to reduce refusal
            # behavior at any tested layer at any tested weight level", while
            # unmatched bulk contrast eliminated refusal at w=0.5 with KL below
            # 0.005. Subtracting the topic subtracts the refusal signal with it.
            # The topic confound the matched design was built to remove turns
            # out to be load bearing.
            bulk = [c["prompt"] for c in tr["corpus"]][: len(pairs)]
            A["bulk_harmless"] = cap(bulk)
            P, N = A["harmful"], A["bulk_harmless"]
            log(f"bulk contrast: {P.shape[0]} harmful vs {N.shape[0]} unmatched harmless")
        elif a.contrast == "matched":
            P, N = A["harmful"], A["harmless"]
            log(f"matched pair contrast: {P.shape[0]} pairs")
        else:
            pos, neg = [], []
            for i, p in enumerate(pairs):
                (pos if p["harmful_refused"] else neg).append(A["harmful"][i])
                (pos if p["harmless_refused"] else neg).append(A["harmless"][i])
            for i, b in enumerate(border):
                (pos if b["prompt_refused"] else neg).append(A["borderline"][i])
            log(f"refuse {len(pos)} comply {len(neg)}")
            if len(pos) < 8 or len(neg) < 8:
                sys.exit("not enough of both classes for a refuse/comply contrast")
            pos, neg = torch.stack(pos), torch.stack(neg)
            P, N = pos, neg
        hm = (A.get("bulk_harmless") if a.contrast == "bulk"
              else A["harmless"]).double().mean(0).float()
        variants = {n: S.build(P, N, k=a.k, paired=(a.contrast == "matched"),
                               harmless_mean=hm if pr else None,
                               protected=protected, protect=pr)
                    for n, pr in (("protected", True), ("plain", False))}
        cond = S.behaviour_directions(A["harmful"], A["harmless"], paired=True)
        for n, v in variants.items():
            save_subspace(out / f"subspace_{n}.pt", v, extra={"model": a.model}, condition=cond)
        torch.save({"condition": cond}, out / "condition.pt")
        meta = {"report": S.report(variants["protected"]), "contrast": a.contrast,
                "clip": a.clip, "k": a.k,
                "base_refusal_harmful": sum(p["harmful_refused"] for p in pairs)/len(pairs),
                "base_refusal_twin": sum(p["harmless_refused"] for p in pairs)/len(pairs),
                "base_over_refusal": sum(b["prompt_refused"] for b in border)/max(len(border),1),
                "n_pairs": len(pairs)}
        mark(out, "extract", meta)
        log(S.report(variants["protected"]))
    ss = variants["protected"]

    # ---------- Stage 0 baseline + layer scan ----------
    sets = build_eval_sets(DATA, split="test", n=a.n_eval)
    # Protected directions are extracted from the train calibration split, so
    # train is burned for measurement. Score on val plus test together: 41
    # items instead of 20, which is about 10 per category instead of 5. At 5 a
    # single flipped item moves a category by 0.20 and the number means
    # nothing.
    held_out = [c for c in D.load(DATA, split="val")["calibration"]] + \
               [c for c in D.load(DATA, split="test")["calibration"]]
    sets.calibration = held_out[: a.cal]
    log(f"calibration scored on {len(sets.calibration)} held out items "
        f"({len(set(c['category'] for c in sets.calibration))} categories)")
    val = D.load(DATA, split="val")
    log(f"caching reference trace on {len(sets.harmless)} harmless prompts")
    trace = teacher_trace(model, tok, sets.harmless, max_new_tokens=a.kl_tokens,
                          top_k=512, batch_size=max(a.batch // 2, 1), device=a.device)
    ev = lambda: evaluate(_EVAL_MODEL[0] or model, tok, sets, trace=trace, judge=judge,
                          max_new_tokens=a.max_new, batch_size=a.batch,
                          kl_batch_size=max(a.batch // 2, 1), device=a.device,
                          with_samples=2, keep_all=a.keep_generations)
    log("baseline")
    baseline = ev()
    log(f"  refusal {baseline.get('refusal_rate'):.3f}  coherent {baseline.get('coherence_rate'):.3f}")

    sc = layerscan.scan(model, tok, ss,
                        [p["harmful"] for p in val["pairs"]][: a.n_eval],
                        [c["prompt"] for c in val["corpus"]][: a.n_eval],
                        batch_size=a.batch, device=a.device,
                        stride=max(1, L // 14), kl_budget=a.kl_budget,
                        quant=a.quant or None)
    bl = sc["best_layer"]
    log(f"layer {bl} chosen at {sc.get('scanned_at')}")

    # ---------- Stage 4: search ----------
    orig = None if hook else snapshot(model)
    # Every precision in --final-quant rounds all decoder linears in place, and
    # restore() only puts back the residual writers. Without this snapshot the
    # q, k, v, gate and up weights stay int4 after the loop, so the saved
    # checkpoint and the healing teacher were both quietly quantized. Found
    # 2026-10-21 when Qwen3-4B scored KL 0.034 in memory and 0.143 cold.
    lin = snapshot_linears(model) if (a.quant or a.final_quant) else None
    if stage_done(out, "search"):
        best = json.loads((out / "search.json").read_text())["best"]
        log(f"stage 4 cached: {best['params']}")
    else:
        log(f"searching {a.trials} trials, scored at {a.quant or 'bf16'}")
        study = run_study(model, tok, variants, sets, trace, baseline, orig,
                          n_trials=a.trials, study_name="ablate",
                          storage=(make_storage("journal", a.study_journal) if a.study_journal
                                   else make_storage("memory")), device=a.device,
                          objective_mode=a.objective, kl_target=a.kl_target,
                          n_startup=a.startup_trials or None,
                          refusal_tol=a.refusal_tol, cal_tol=a.cal_tol,
                          batch_size=a.batch, kl_batch_size=max(a.batch // 2, 1),
                          judge=judge, out_dir=out / "trials", seed=a.seed,
                          grid=(rank1_grid() if a.grid == "rank1" else None),
                          quant=a.quant or None, linears=lin, edit_mode=a.edit_mode)
        best = best_plan(study, a.refusal_tol, a.cal_tol)
        mark(out, "search", {"best": best, "pareto": pareto(study),
                             "n_trials": len(study.trials)})
        log(f"best {best['value']:.4f}  {best['params']}")

    p = best["params"]
    _center = p["center_frac"] * (L - 1)
    plan = EditPlan(center=_center,
                    width=p.get("width_frac", 0.2) * (L - 1) if "width_frac" in p else p.get("width", 3.5),
                    peak=p["peak"], kernel=p["kernel"],
                    kinds={"both": ("attn_out", "mlp_out"), "mlp_only": ("mlp_out",),
                           "attn_only": ("attn_out",)}[p["kinds"]],
                    norm_restore=p["norm_restore"],
                    source_layer=int(round(_center)) if p.get("broadcast") else None)
    chosen = variants.get(p.get("subspace", "protected"), ss).with_rank(p["k"])

    # ---------- Stage 3 + 6 ----------
    results = {"baseline": baseline}
    if hook:
        # scored exactly as searched; the weights stay the base model's
        with HookEdit(model, chosen, plan) as he:
            rec = he.record
            log(f"hooked: {rec['n_edited']} matrices")
            results["edited"] = ev()
        surv = {}
    else:
        restore(model, orig)
        if lin: restore_linears(model, lin)
        rec = apply_plan(model, chosen, plan, originals=orig)
        log(f"applied: {rec['n_edited']} matrices")
        results["edited"] = ev()

        qkw = PRESETS.get(a.quant, {"bits": 4, "group_size": 128})
        surv = survival_report(model, orig, residual_writers(model),
                               bits=qkw["bits"], group_size=qkw["group_size"])
        log(f"edit survival at {a.quant or 'int4-g128'}: retained "
            f"{surv.get('retained_mean', float('nan')):.3f}  "
            f"ratio {surv.get('ratio_overall', float('nan')):.2f}")

    # each precision starts from unquantized weights plus a fresh edit, so
    # quantization error never compounds across the loop
    for pres in [x.strip() for x in a.final_quant.split(",") if x.strip()]:
        if pres not in PRESETS:
            log(f"  unknown preset {pres}, skipping"); continue
        if lin is not None:
            restore_linears(model, lin)
        restore(model, orig)
        apply_plan(model, chosen, plan, originals=orig)
        st = quantize_model_(model, **PRESETS[pres])
        log(f"edited@{pres}  rel weight rmse {st['rel_weight_rmse']:.5f}")
        results[f"edited@{pres}"] = ev()
        results[f"edited@{pres}"]["_quant"] = st

    # leave the model holding the bf16 edit, which is what gets saved
    if lin is not None:
        restore_linears(model, lin)
    if not hook:
        restore(model, orig)
        apply_plan(model, chosen, plan, originals=orig)

    for k in [x for x in results if x != "baseline"]:
        results[k]["_objective"] = objective(baseline, results[k])

    if lin is not None:
        # everything the edit does not write must still be the untouched model
        # writer names and module names use different prefixes, so match by module
        writers = {id(w.module) for w in residual_writers(model)}
        rest = [(n, m) for n, m in _targets(model, SKIP) if id(m) not in writers and n in lin]
        bad = [n for n, m in rest if not torch.equal(m.weight.detach().cpu(), lin[n])]
        if bad:
            sys.exit(f"non edited linears changed after the quant loop: {bad[:4]} ({len(bad)})")
        log(f"verified {len(rest)} non edited linears untouched")

    if a.save_weights:
        results["edited:ckpt"] = ship_and_reload(model, tok, out / "ckpt-edited", a, ev)


    # ---------- Stage 5: self distillation healing ----------
    healing = {"steps": a.heal_steps, "qat": a.qat}
    if a.heal_steps > 0:
        from ablate.quant import detach_fake_quant
        log(f"stage 5: healing {a.heal_steps} steps"
            f"{', quantization aware' if a.qat else ''}")
        # the teacher is the untouched model, so build its trace before the edit
        tr = D.load(DATA, split="train")
        corpus = [c["prompt"] for c in tr["corpus"]][: a.heal_corpus]
        if a.heal_probes:
            corpus += [c["probe"] for c in D.load(DATA, split="train")["calibration"]]
        if lin is not None:
            restore_linears(model, lin)
        restore(model, orig)
        log(f"  teacher trace over {len(corpus)} harmless prompts")
        heal_trace = teacher_trace(model, tok, corpus, max_new_tokens=a.kl_tokens,
                                   top_k=256, batch_size=max(a.batch // 2, 1),
                                   device=a.device)
        # then put the edit back and train through it
        apply_plan(model, chosen, plan, originals=orig)
        if a.qat:
            n = attach_qat(model, **{k: v for k, v in PRESETS[a.quant].items()
                                     if k in ("bits", "group_size")})                 if a.quant in PRESETS else attach_qat(model)
            log(f"  quantization aware on {n} linears")
        lm = attach_lora(model, r=a.lora_r, alpha=2 * a.lora_r)
        hist = heal(lm, heal_trace, steps=a.heal_steps, lr=a.heal_lr,
                    batch_size=max(a.batch // 8, 1), grad_accum=4,
                    device=a.device, log_every=max(a.heal_steps // 5, 1))
        healing.update({k: v for k, v in hist.items() if k != "history"})
        log(f"  KL {hist['loss_first']:.4f} -> {hist['loss_last']:.4f}")
        # detach before merging. peft writes the merged weight back through
        # the parameter, and a parametrized weight has no right_inverse to
        # write through, so merging while still attached fails. Detaching with
        # leave_parametrized False also restores full precision weights, so the
        # LoRA merges into bf16 and quantization happens once, at the end.
        if a.qat:
            detach_fake_quant(lm)
        model = lm.merge_and_unload()
        # Distilling toward the untouched teacher re-teaches refusal: gemma-3-1b
        # bulk went 0.188 -> 0.604 on 2026-10-21. The LoRA delta on the writers
        # is free to point back along the subspace, so project it out again with
        # the same plan, starting from the merged weights rather than originals.
        if a.heal_reproject:
            results["healed:raw"] = ev()
            rp = apply_plan(model, chosen, plan, originals=None)
            healing["reprojected"] = rp["n_edited"]
            log(f"  reprojected {rp['n_edited']} merged writers")
        results["healed"] = ev()
        for pres in [x.strip() for x in a.final_quant.split(",") if x.strip()]:
            if pres not in PRESETS:
                continue
            hl = snapshot_linears(model)
            quantize_model_(model, **PRESETS[pres])
            results[f"healed@{pres}"] = ev()
            restore_linears(model, hl)
        if a.save_weights:
            results["healed:ckpt"] = ship_and_reload(model, tok, out / "ckpt-healed", a, ev)
        for k in [x for x in results if x.startswith("healed")]:
            results[k]["_objective"] = objective(baseline, results[k])

    card = {"name": out.name, "base_model": a.model, "built": time.strftime("%Y-%m-%d"),
            "extraction": meta, "plan": asdict(plan), "layer_scan": sc,
            "search": best, "survival": {k: v for k, v in surv.items() if k != "per_tensor"},
            "results": results, "healing": healing,
            "edit_mode": a.edit_mode, "held_as": a.bnb or a.dtype,
            "data": {"pairs": a.pairs, "n_eval": a.n_eval, "cal": a.cal},
            "seconds": round(time.time() - t0, 1)}
    # generations go beside the card, not inside it, so the card stays readable
    if a.keep_generations:
        gen_dump = {k: v.pop("_generations") for k, v in results.items()
                    if isinstance(v, dict) and "_generations" in v}
        (out / "generations.json").write_text(json.dumps(gen_dump, indent=2, default=str))
        log(f"stored generations for {len(gen_dump)} variants")
    write_model_card(out, card)
    log(f"\nwrote {out}/README.md and ablation_card.json  [{card['seconds']:.0f}s]")
    for k, r in results.items():
        log(f"  {k:<22} refusal {r.get('refusal_rate', float('nan')):.3f}  "
            f"coherent {r.get('coherence_rate', float('nan')):.3f}  "
            f"KL {r.get('kl_mean', float('nan')):.4f}")


if __name__ == "__main__":
    main()
