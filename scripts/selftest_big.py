"""Self test for the large model path: HookEdit and the stream baker must both
reproduce apply_plan exactly. Random tiny model, CPU, no downloads.

  python scripts/selftest_big.py
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from ablate import subspace as S
from ablate.apply import EditPlan, apply_plan
from ablate.hookedit import HookEdit
from ablate.modelio import n_layers, residual_writers
from ablate.pipeline import save_subspace
from ablate.streambake import bake, read_header, resolve_writers

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + detail) if detail else ''}", flush=True)


def tiny(dtype=torch.float32, bias=False):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=160,
                      num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    if bias:     # no stock family biases its writers; force one to cover that branch
        for w in residual_writers(m):
            w.module.bias = torch.nn.Parameter(torch.randn(w.module.out_features) * 0.05)
    m.requires_grad_(False)
    return m.to(dtype)


def random_subspace(L, H, k=3):
    torch.manual_seed(1)
    basis = torch.stack([torch.linalg.qr(torch.randn(H, k))[0].T for _ in range(L + 1)])
    return S.Subspace(basis=basis, variance_ratio=torch.ones(L + 1, k) / k,
                      separation=torch.ones(L + 1), rank=torch.full((L + 1,), k))


PLANS = {
    "uniform column broadcast": dict(center=2.5, width=3.0, peak=0.97, kernel="uniform",
                                     norm_restore="column", source_layer=3),
    "gaussian none per layer": dict(center=3.0, width=1.5, peak=1.0, kernel="gaussian",
                                    norm_restore="none"),
    "band column mlp only": dict(center=2.0, width=1.0, peak=0.8, kernel="band",
                                 norm_restore="column", kinds=("mlp_out",)),
}


def moe_checks(ids):
    """Mixture of experts (the GLM-5 layout): two independent routes to the same
    edit must agree. The hook projects each block's output; the baker projects
    every expert's down projection on disk."""
    from transformers import AutoModelForCausalLM, DeepseekV3Config, DeepseekV3ForCausalLM
    print("mixture of experts: hook on the block vs per expert stream bake")
    cfg = DeepseekV3Config(vocab_size=512, hidden_size=64, intermediate_size=160,
                           moe_intermediate_size=48, num_hidden_layers=6, num_attention_heads=4,
                           num_key_value_heads=4, n_routed_experts=8, n_shared_experts=1,
                           num_experts_per_tok=2, n_group=1, topk_group=1, first_k_dense_replace=2, kv_lora_rank=16,
                           q_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16,
                           max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    base = DeepseekV3ForCausalLM(cfg).eval()
    base.requires_grad_(False)
    L, H = n_layers(base), cfg.hidden_size
    ss = random_subspace(L, H)
    ws = residual_writers(base)
    check("experts resolved as block writers",
          sum(w.block for w in ws) == 4 and sum(w.kind == "attn_out" for w in ws) == 6
          and sum(w.kind == "mlp_out" for w in ws) == 6, f"{len(ws)} writers, {sum(w.block for w in ws)} blocks")
    plan_none = EditPlan(center=3.0, width=3.0, peak=0.9, kernel="uniform", norm_restore="none",
                         source_layer=3)
    try:
        apply_plan(copy.deepcopy(base), ss, plan_none)
        check("weight edit refuses a mixture of experts", False)
    except NotImplementedError:
        check("weight edit refuses a mixture of experts", True)
    try:
        with HookEdit(base, ss, EditPlan(center=3.0, width=3.0, peak=0.9, kernel="uniform")):
            pass
        check("hook refuses column restore on a block", False)
    except NotImplementedError:
        check("hook refuses column restore on a block", True)
    ref = base(input_ids=ids).logits
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base.save_pretrained(td / "base", safe_serialization=True, max_shard_size="300KB")
        save_subspace(td / "subspace_plain.pt", ss)
        names = [n for s in sorted((td / "base").glob("*.safetensors")) for n in read_header(s)[0]]
        res = resolve_writers(names, cfg.num_hidden_layers)
        check("stream resolver finds every expert",
              sum(k == "mlp_out" for _, k in res.values()) == 2 + 4 * 9
              and sum(k == "attn_out" for _, k in res.values()) == 6, f"{len(res)} tensors")
        for label, plan in (("uniform broadcast", plan_none),
                            ("gaussian per layer mlp only",
                             EditPlan(center=3.5, width=1.2, peak=1.0, kernel="gaussian",
                                      norm_restore="none", kinds=("mlp_out",)))):
            for k in (1, 3):
                man = bake(td / "base", td / f"out_{label[:7]}_{k}", td / "subspace_plain.pt", plan,
                           k=k, log=lambda *_: None)
                baked = AutoModelForCausalLM.from_pretrained(td / f"out_{label[:7]}_{k}").eval()
                want = baked(input_ids=ids).logits
                with HookEdit(base, ss.with_rank(k), plan) as he:
                    got = base(input_ids=ids).logits
                moved, err = float((want - ref).abs().max()), float((got - want).abs().max())
                check(f"{label} k={k}", err < 2e-4 * max(1.0, moved) and moved > 1e-3,
                      f"edit moved logits {moved:.3f}, hook vs bake {err:.2e}, "
                      f"{man['n_edited']} tensors baked, {he.record['n_edited']} hooks")
    check("hooks removed, model unchanged", torch.equal(base(input_ids=ids).logits, ref))


def stream_checks():
    """Candidate scoring for models that cannot generate: the hook scored first
    token metrics must equal the same metrics on a weight edited copy, and the
    payload must be something the baker accepts."""
    import json
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from ablate.streambake import plan_from_search
    from ablate.streamsearch import _first_token, candidates, score_candidates
    from ablate.capture import refusal_token_ids
    print("stream search: first token scoring through hooks")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok.padding_side = "left"
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=160, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=True)).eval()
    model.requires_grad_(False)
    L = n_layers(model)
    ss = random_subspace(L, 64, k=4)
    harmful = [f"request number {i} about a locked door" for i in range(6)]
    harmless = [f"question number {i} about baking bread" for i in range(5)]
    cands = candidates(L, 3, ks=(1, 4), kernels=("uniform", "band"))
    with tempfile.TemporaryDirectory() as td:
        res = score_candidates(model, tok, {"plain": ss}, harmful, harmless, cands, batch_size=4,
                               kl_budget=1e9, out_dir=Path(td) / "cands", log=lambda *_: None)
        again = score_candidates(model, tok, {"plain": ss}, harmful, harmless, cands, batch_size=4,
                                 kl_budget=1e9, out_dir=Path(td) / "cands", log=lambda *_: None)
    check("four candidates scored, resume returns the same rows",
          len(res["candidates"]) == 4 and again["candidates"] == res["candidates"]
          and json.loads(json.dumps(res))["best"]["params"] in cands)
    rid = refusal_token_ids(tok)
    base = _first_token(model, tok, harmful, harmless, rid, 4)
    worst = 0.0
    for row in res["candidates"]:
        p = row["params"]
        m = copy.deepcopy(model)
        apply_plan(m, ss.with_rank(p["k"]), plan_from_search(p, L))
        w = _first_token(m, tok, harmful, harmless, rid, 4, ref=base["_harmless_lp"])
        worst = max(worst, abs(w["ft_kl"] - row["ft_kl"]), abs(w["refusal_lp"] - row["refusal_lp"]))
    kls = [r["ft_kl"] for r in res["candidates"]]
    check("hook scores equal weight edit scores", worst < 1e-4 and min(kls) > 0,
          f"max metric difference {worst:.1e}, first token KL {min(kls):.4f} to {max(kls):.4f}")
    best = max(res["candidates"], key=lambda r: (r["drop"], -r["ft_kl"]))
    check("best is the largest drop within budget", res["best"]["params"] == best["params"])


def main():
    ids = torch.randint(0, 512, (4, 24), generator=torch.Generator().manual_seed(2))
    for bias in (False, True):
        base = tiny(bias=bias)
        L, H = n_layers(base), base.config.hidden_size
        ss = random_subspace(L, H)
        ref = base(input_ids=ids).logits
        print(f"hook edit vs weight edit (writer bias {bias})")
        for name, kw in PLANS.items():
            plan = EditPlan(**kw)
            for k in (1, 3):
                sk = ss.with_rank(k)
                m = copy.deepcopy(base)
                rec = apply_plan(m, sk, plan)
                want = m(input_ids=ids).logits
                with HookEdit(base, sk, plan) as he:
                    got = base(input_ids=ids).logits
                moved = float((want - ref).abs().max())
                err = float((got - want).abs().max())
                check(f"{name} k={k}", err < 2e-4 * max(1.0, moved) and moved > 1e-3
                      and he.record["n_edited"] == rec["n_edited"],
                      f"edit moved logits {moved:.3f}, hook vs weights {err:.2e}, {rec['n_edited']} writers")
        after = base(input_ids=ids).logits
        check("hooks removed, model unchanged", torch.equal(after, ref))

    print("stream bake vs weight edit (bf16 shards)")
    base = tiny(torch.bfloat16)
    L, H = n_layers(base), base.config.hidden_size
    ss = random_subspace(L, H)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base.save_pretrained(td / "base", safe_serialization=True, max_shard_size="200KB")
        shards = sorted((td / "base").glob("*.safetensors"))
        names = [n for s in shards for n in read_header(s)[0]]
        res = resolve_writers(names)
        sd_names = {n for n, _ in base.named_parameters()}
        mods = {id(w.module): (w.layer, w.kind) for w in residual_writers(base)}
        truth = {f"{n}.weight": mods[id(mod)] for n, mod in base.named_modules() if id(mod) in mods}
        check("writer resolution matches modelio", res == truth,
              f"{len(res)} tensors over {len(shards)} shards")
        save_subspace(td / "subspace_plain.pt", ss)
        for name, kw in PLANS.items():
            plan = EditPlan(**kw)
            m = copy.deepcopy(base)
            apply_plan(m, ss.with_rank(2), plan)
            out = td / ("out_" + name.replace(" ", "_"))
            man = bake(td / "base", out, td / "subspace_plain.pt", plan, k=2, log=lambda *_: None)
            from transformers import AutoModelForCausalLM
            got = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16)
            want, have, orig = m.state_dict(), got.state_dict(), base.state_dict()
            worst = max(float((want[n].float() - have[n].float()).abs().max()) for n in want)
            same = all(torch.equal(orig[n], have[n]) for n in orig
                       if n not in {e["tensor"] for e in man["edited"]})
            changed = sum(not torch.equal(orig[e["tensor"]], have[e["tensor"]]) for e in man["edited"])
            check(f"{name}", worst == 0.0 and same and changed == man["n_edited"] and sd_names >= set(have) - {"lm_head.weight"},
                  f"max abs diff {worst:.1e}, {changed}/{man['n_edited']} writers changed, rest bit identical {same}")

    moe_checks(ids)
    stream_checks()
    print(f"\n{len(OK)} ok, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
