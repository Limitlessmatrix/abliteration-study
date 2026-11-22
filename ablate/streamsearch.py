"""Stage 4 for models that cannot afford generation.

The Optuna search scores every trial with generation and a teacher forced KL,
dozens of forward passes per trial. On a model held partly on disk each pass
re-reads the offloaded weights, so that search is out of reach. This scores a
short list of candidate plans with ONE forward pass each, through HookEdit:

  refusal_lp   log probability mass on refusal openers at the first generated
               position, on held out harmful prompts (lower is less refusal)
  ft_kl        KL(base || edited) of the first token distribution on held out
               harmless prompts

That is the layer scan's metric, applied to whole plans. It is a proxy: it says
which candidate moves the opening of the reply furthest for the least drift,
not what the refusal rate is. The real numbers come from scoring the baked
artifact as served. The candidates default to the shape every finished run of
ours converged on: one direction set broadcast from the best layer, both writer
kinds, a uniform or wide band kernel, peak near 1.

search.json is written with the same keys the Optuna stage writes, so
scripts/bake_stream.py takes either.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import torch

from .capture import first_token_stats, refusal_token_ids
from .hookedit import HookEdit
from .modelio import has_block_writers, n_layers
from .streambake import plan_from_search


def candidates(L: int, best_layer: int, ks=(1, 2, 4), peaks=(1.0,),
               kernels=("uniform", "band"), variants=("plain",),
               norms=("column",), band_width_frac: float = 0.35) -> list[dict]:
    out = []
    for kern, k, peak, var, norm in itertools.product(kernels, ks, peaks, variants, norms):
        out.append({"k": int(k), "center_frac": best_layer / max(L - 1, 1),
                    "width_frac": band_width_frac, "peak": float(peak), "kernel": kern,
                    "kinds": "both", "norm_restore": norm, "subspace": var, "broadcast": True})
    return out


@torch.no_grad()
def _first_token(model, tok, harmful, harmless, rid, batch_size, ref=None) -> dict:
    """Both prompt sets in as few batches as allowed, so an offloaded model is
    read once per call when batch_size covers them."""
    st = first_token_stats(model, tok, harmful + harmless, rid, batch_size=batch_size)
    lp = st["_logprobs"]
    r = torch.tensor(rid)
    h, c = lp[: len(harmful)], lp[len(harmful):]
    res = {"refusal_lp": float(torch.logsumexp(h[:, r], dim=-1).mean()),
           "refusal_lp_harmless": float(torch.logsumexp(c[:, r], dim=-1).mean()),
           "_harmless_lp": c}
    if ref is not None:
        res["ft_kl"] = float((ref.exp() * (ref - c)).sum(-1).clamp(min=0).mean())
    return res


def score_candidates(model, tok, variants: dict, harmful: list[str], harmless: list[str],
                     cands: list[dict], batch_size: int = 64, kl_budget: float = 0.1,
                     out_dir: Path | None = None, log=print) -> dict:
    """variants maps a name to a Subspace. Returns the search.json payload."""
    L = n_layers(model)
    rid = refusal_token_ids(tok)
    if has_block_writers(model):
        cands = [dict(c, norm_restore="none") for c in cands]
    cands = [c for i, c in enumerate(cands) if c not in cands[:i]]
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    base_f = out_dir / "baseline.pt" if out_dir else None
    if base_f and base_f.exists():
        base = torch.load(base_f, map_location="cpu", weights_only=True)
        log("baseline cached")
    else:
        base = _first_token(model, tok, harmful, harmless, rid, batch_size)
        if base_f:
            torch.save(base, base_f)
    ref = base["_harmless_lp"]
    log(f"baseline  refusal_lp {base['refusal_lp']:.3f} on harmful, "
        f"{base['refusal_lp_harmless']:.3f} on harmless")

    rows = []
    for i, p in enumerate(cands):
        f = out_dir / f"cand_{i:03d}.json" if out_dir else None
        if f and f.exists() and json.loads(f.read_text())["params"] == p:
            row = json.loads(f.read_text())          # an interrupted run resumes here
        else:
            ss = variants.get(p["subspace"]) or next(iter(variants.values()))
            with HookEdit(model, ss.with_rank(p["k"]), plan_from_search(p, L)) as he:
                r = _first_token(model, tok, harmful, harmless, rid, batch_size, ref=ref)
            row = {"params": p, "refusal_lp": r["refusal_lp"], "ft_kl": r["ft_kl"],
                   "drop": base["refusal_lp"] - r["refusal_lp"], "n_edited": he.record["n_edited"]}
            if f:
                f.write_text(json.dumps(row, indent=1))
        rows.append(row)
        log(f"cand {i:02d}  drop {row['drop']:+.3f}  ft_kl {row['ft_kl']:.4f}  "
            f"k {p['k']} {p['kernel']} peak {p['peak']} {p['subspace']} {p['norm_restore']}")

    ok = [r for r in rows if r["ft_kl"] <= kl_budget]
    within = bool(ok)
    best = max(ok, key=lambda r: (r["drop"], -r["ft_kl"])) if ok else min(rows, key=lambda r: r["ft_kl"])
    if not within:
        log(f"no candidate within the KL budget {kl_budget}; taking the lowest KL one")
    return {"best": {"params": best["params"], "value": -best["drop"],
                     "terms": {"refusal_lp_drop": best["drop"], "first_token_kl": best["ft_kl"]},
                     "variant": best["params"]["subspace"], "n_edited": best["n_edited"]},
            "candidates": rows, "n_trials": len(rows), "scored_by": "first_token",
            "kl_budget": kl_budget, "within_budget": within,
            "baseline": {k: v for k, v in base.items() if not k.startswith("_")}}
