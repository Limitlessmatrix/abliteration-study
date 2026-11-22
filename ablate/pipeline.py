"""End to end drivers: extract a subspace, apply a plan, score the result.

Stage boundaries are kept as separate callables so the Optuna search in Stage 4
can reuse one extraction across many plans, which is where almost all the time
goes.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import calibration as CAL
from . import datasets as D
from . import subspace as S
from .apply import EditPlan, apply_plan, naive_plan, snapshot
from .capture import capture_residuals, teacher_trace
from .evalharness import EvalSets, evaluate, objective
from .modelio import load


@dataclass
class ExtractConfig:
    contrast: str = "refuse_vs_comply"   # or harmful_vs_harmless
    k: int = 2
    protect: bool = True
    orthogonalise_harmless_mean: bool = True
    max_pairs: int = 256
    batch_size: int = 16
    label_behaviour: bool = True
    split: str = "train"


@dataclass
class Artifacts:
    subspace: S.Subspace
    variants: dict = field(default_factory=dict)   # name -> Subspace
    acts: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)
    condition: torch.Tensor | None = None     # [L+1, H] harmfulness recognition


def _acts(model, tok, prompts, bs, device):
    return capture_residuals(model, tok, prompts, batch_size=bs, device=device)


def extract(model, tok, data_root: Path, cfg: ExtractConfig,
            device: str = "cuda", verbose: bool = True) -> Artifacts:
    sets = D.load(data_root, split=cfg.split)
    pairs = sets["pairs"][: cfg.max_pairs]
    cal = sets["calibration"]
    border = sets["borderline"][: cfg.max_pairs]
    if not pairs:
        raise RuntimeError(f"no pairs in split {cfg.split}; run the data build first")

    t0 = time.time()
    if cfg.label_behaviour:
        if verbose:
            print(f"labelling behaviour on {len(pairs)} pairs and {len(border)} borderline", flush=True)
        D.label_behaviour(model, tok, pairs, "harmful", device=device, batch_size=cfg.batch_size)
        D.label_behaviour(model, tok, pairs, "harmless", device=device, batch_size=cfg.batch_size)
        D.label_behaviour(model, tok, border, "prompt", device=device, batch_size=cfg.batch_size)

    # A pair only carries refusal information if the model actually refuses the
    # harmful side and actually answers the twin. The generated twin is not
    # always benign: "commit suicide" turns into "commit a heart attack", which
    # gets refused too. Paired contrast needs those dropped; the unpaired
    # refuse against comply contrast can keep them, because it routes every
    # prompt by what the model did rather than by which column it came from.
    n_raw = len(pairs)
    if cfg.label_behaviour and cfg.contrast == "harmful_vs_harmless":
        pairs = [p for p in pairs
                 if p.get("harmful_refused") and not p.get("harmless_refused")]
        if verbose:
            print(f"behavioural gate kept {len(pairs)}/{n_raw} pairs", flush=True)
        if len(pairs) < 16:
            raise RuntimeError(
                f"only {len(pairs)} clean pairs survived the behavioural gate. "
                "Either the model barely refuses this benchmark, or the twins "
                "are landing harmful. Inspect data/sets/pairs.jsonl.")
    harmful = [p["harmful"] for p in pairs]
    harmless = [p["harmless"] for p in pairs]

    if verbose:
        print(f"capturing residuals: {len(harmful)} harmful, {len(harmless)} harmless, "
              f"{len(cal)} calibration pairs, {len(border)} borderline", flush=True)
    A = {
        "harmful": _acts(model, tok, harmful, cfg.batch_size, device),
        "harmless": _acts(model, tok, harmless, cfg.batch_size, device),
        "cal_probe": _acts(model, tok, [c["probe"] for c in cal], cfg.batch_size, device),
        "cal_control": _acts(model, tok, [c["control"] for c in cal], cfg.batch_size, device),
        "borderline": _acts(model, tok, [b["prompt"] for b in border], cfg.batch_size, device),
    }

    # Stage 2.3 protected directions, one per calibration behaviour
    protected = {}
    for cat in CAL.CATEGORIES:
        idx = [i for i, c in enumerate(cal) if c["category"] == cat]
        if len(idx) < 4:
            continue
        protected[cat] = S.behaviour_directions(A["cal_probe"][idx], A["cal_control"][idx])

    harmless_mean = A["harmless"].double().mean(0).float() if cfg.orthogonalise_harmless_mean else None

    # Stage 2.4 choose what the contrast actually is
    if cfg.contrast == "harmful_vs_harmless":
        pos, neg, paired = A["harmful"], A["harmless"], True
    elif cfg.contrast == "refuse_vs_comply":
        pos_rows, neg_rows = [], []
        for i, p in enumerate(pairs):
            (pos_rows if p.get("harmful_refused") else neg_rows).append(A["harmful"][i])
            (pos_rows if p.get("harmless_refused") else neg_rows).append(A["harmless"][i])
        for i, b in enumerate(border):
            (pos_rows if b.get("prompt_refused") else neg_rows).append(A["borderline"][i])
        if len(pos_rows) < 8 or len(neg_rows) < 8:
            raise RuntimeError(
                f"refuse_vs_comply needs both classes; got refuse={len(pos_rows)} "
                f"comply={len(neg_rows)}. The model may not refuse this set at all.")
        pos, neg, paired = torch.stack(pos_rows), torch.stack(neg_rows), False
    else:
        raise ValueError(cfg.contrast)

    # Both variants come off the same capture, so the plain one is free and the
    # Stage 4 search can treat protection as just another dimension.
    variants = {}
    for name, do_protect in (("protected", True), ("plain", False)):
        v = S.build(pos, neg, k=cfg.k, paired=paired,
                    harmless_mean=harmless_mean if do_protect else None,
                    protected=protected, protect=do_protect)
        v.meta.update({"contrast": cfg.contrast, "n_pos": int(pos.shape[0]),
                       "n_neg": int(neg.shape[0]), "variant": name,
                       "seconds": round(time.time() - t0, 1)})
        variants[name] = v
    ss = variants["protected" if cfg.protect else "plain"]

    # The gate for conditional ablation runs on harmfulness recognition, which
    # Zhao et al. 2025 show is encoded separately from the refusal decision.
    # Extracting it from the same capture costs nothing and keeps the gate
    # independent of the subspace being removed.
    condition = S.behaviour_directions(A["harmful"], A["harmless"], paired=True)

    labels = {
        "harmful_refused": [bool(p.get("harmful_refused")) for p in pairs],
        "harmless_refused": [bool(p.get("harmless_refused")) for p in pairs],
        "borderline_refused": [bool(b.get("prompt_refused")) for b in border],
    }
    diag = {
        "pairs_raw": n_raw,
        "pairs_used": len(pairs),
        "base_refusal_rate_harmful": sum(labels["harmful_refused"]) / max(len(pairs), 1),
        "base_refusal_rate_twin": sum(labels["harmless_refused"]) / max(len(pairs), 1),
        "base_over_refusal_rate": sum(labels["borderline_refused"]) / max(len(border), 1),
        "n_pairs": len(pairs),
        "pairs_strict": sum(p.get("tier") == "strict" for p in pairs),
    }
    if verbose:
        print(S.report(ss), flush=True)
        print(f"base refusal: harmful {diag['base_refusal_rate_harmful']:.2f}  "
              f"twin {diag['base_refusal_rate_twin']:.2f}  "
              f"borderline {diag['base_over_refusal_rate']:.2f}", flush=True)

    return Artifacts(subspace=ss, variants=variants, acts=A, labels=labels,
                     diag=diag, condition=condition)


def probe_separation(art: Artifacts, layer: int | None = None) -> dict:
    """Stage 2.4 check: does the activation carry the refusal decision, or only
    harmfulness recognition? Fit a probe for each and compare AUC at the layer
    the subspace was chosen at."""
    ss = art.subspace
    layer = layer if layer is not None else ss.best_layer()
    A, L = art.acts, art.labels
    n_pairs = A["harmful"].shape[0]

    acts = torch.cat([A["harmful"], A["harmless"], A["borderline"]], dim=0)
    refused = torch.tensor([int(x) for x in L["harmful_refused"]]
                           + [int(x) for x in L["harmless_refused"]]
                           + [int(x) for x in L["borderline_refused"]])
    harmfulness = torch.tensor([1] * n_pairs + [0] * n_pairs
                               + [0] * A["borderline"].shape[0])
    try:
        return {
            "layer": layer,
            "auc_refusal_decision": S.probe_auc(acts, refused, layer),
            "auc_harmfulness": S.probe_auc(acts, harmfulness, layer),
            "n": int(acts.shape[0]),
            "n_refused": int(refused.sum()),
        }
    except ImportError:
        return {"layer": layer, "error": "scikit-learn not installed"}


def build_eval_sets(data_root: Path, split: str = "test", n: int = 64) -> EvalSets:
    sets = D.load(data_root, split=split)
    return EvalSets(
        harmful=[p["harmful"] for p in sets["pairs"]][:n],
        harmless=[c["prompt"] for c in sets["corpus"]][:n],
        borderline=[b["prompt"] for b in sets["borderline"]][:n],
        calibration=sets["calibration"],
    )


def save_subspace(path: Path, ss: S.Subspace, extra: dict | None = None,
                  condition: torch.Tensor | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "basis": ss.basis, "variance_ratio": ss.variance_ratio,
        "separation": ss.separation, "rank": ss.rank,
        "protected_overlap": ss.protected_overlap, "meta": ss.meta,
        "condition": condition, "extra": extra or {},
    }, path)


def load_subspace(path: Path) -> S.Subspace:
    d = torch.load(path, map_location="cpu", weights_only=False)
    ss = S.Subspace(basis=d["basis"], variance_ratio=d["variance_ratio"],
                    separation=d["separation"], rank=d["rank"],
                    protected_overlap=d.get("protected_overlap", {}),
                    meta=d.get("meta", {}))
    ss.meta["_condition"] = d.get("condition")
    return ss


def load_condition(path: Path) -> torch.Tensor | None:
    return torch.load(path, map_location="cpu", weights_only=False).get("condition")
