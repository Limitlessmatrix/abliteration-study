"""Stage 4: multi objective search over the edit.

One extraction serves the whole sweep. The trial picks the subspace variant,
the rank actually used, the kernel geometry and which matrices get touched,
then scores on validation. The expensive part is the eval, not the edit, so the
batch sizes here matter more than anything else.

Storage note: SQLite on a network filesystem corrupts under concurrent writers. The default
here is Optuna's journal file backend, which is safe on a shared filesystem.
For many parallel workers, pass a Postgres URL (make_storage("postgres", url=...)).
"""
from __future__ import annotations

import os

import json
from dataclasses import asdict
from pathlib import Path

from .apply import EditPlan, apply_plan, restore
from .evalharness import evaluate, objective
from .hookedit import HookEdit
from .quant import PRESETS, quantize_model_, restore_linears, snapshot_linears


def make_storage(kind: str = "journal", path: str | None = None, url: str | None = None):
    import optuna
    if kind == "journal":
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        p = path or os.environ.get("ABLATE_JOURNAL", "runs/optuna.log")
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        return JournalStorage(JournalFileBackend(p))
    if kind == "postgres":
        if not url:
            raise ValueError("postgres storage needs a url")
        return url
    if kind == "memory":
        return None
    raise ValueError(kind)


# The rank ceiling is the top of the extracted subspace, so it has to track
# whatever --k the extract stage ran with. ABLATE_K_MAX lets a job raise both
# together; asking for more rank than the basis holds silently caps the edit.
_K_MAX = int(os.environ.get("ABLATE_K_MAX", "4"))

SPACE = {
    "k": (1, _K_MAX),
    "center_frac": (0.20, 0.85),
    "width_frac": (0.05, 0.60),   # as a fraction of depth, not an absolute
    "peak": (0.5, 1.0),
}


def suggest_plan(trial, n_layers: int, allow_uniform: bool = True,
                 norm_choices=("column", "none")) -> tuple[EditPlan, int, str]:
    """Sample an edit plan.

    Two things here are set by what published working recipes actually do
    rather than by what reads as principled. Width is a fraction of depth, not
    an absolute number of layers: Grimjim's gemma-3-12b configuration ablates
    layers 11 to 41 of 48, so roughly two thirds of the stack, and an absolute
    width range tuned on a small model cannot express that on a large one. And
    peak starts at 0.5 rather than 0.3, because a Gaussian that peaks at 1.0
    still averages well under half strength across the layers it touches, which
    is how an edit ends up looking surgical and doing nothing.

    source_layer broadcasts one measured direction across the whole band, which
    is what the reference implementations do. Per layer directions are the more
    principled story and remain available; this lets the search decide.
    """
    k = trial.suggest_int("k", *SPACE["k"])
    center = trial.suggest_float("center_frac", *SPACE["center_frac"]) * (n_layers - 1)
    width = trial.suggest_float("width_frac", *SPACE["width_frac"]) * (n_layers - 1)
    peak = trial.suggest_float("peak", *SPACE["peak"])
    kernel = trial.suggest_categorical(
        "kernel", ["band", "gaussian", "triangular"] + (["uniform"] if allow_uniform else []))
    kinds_choice = trial.suggest_categorical("kinds", ["both", "mlp_only", "attn_only"])
    kinds = {"both": ("attn_out", "mlp_out"), "mlp_only": ("mlp_out",),
             "attn_only": ("attn_out",)}[kinds_choice]
    norm_restore = trial.suggest_categorical("norm_restore", list(norm_choices))
    variant = trial.suggest_categorical("subspace", ["protected", "plain"])
    broadcast = trial.suggest_categorical("broadcast", [True, False])
    source = int(round(center)) if broadcast else None
    return EditPlan(center=center, width=width, peak=peak, kernel=kernel,
                    kinds=kinds, norm_restore=norm_restore,
                    source_layer=source), k, variant


def rank1_grid(norm_choices=("column", "none")) -> list[dict]:
    """Fixed first trials for the search (2026-11-13). 24 rank 1 band edits covering the region every
    low KL solution on Qwen3 came from: centre 0.50 / 0.60 of depth, width 0.15 / 0.25 / 0.40, peak 0.8 / 1.0,
    plain and protected subspace, attention and MLP writers, one broadcast direction, no norm restore.

    Why: TPE's first ~45 trials are random, and on Qwen3-8B two seeds in three had no usable trial by trial 40
    while one seed in three never found the narrow band at all (search KL 0.0083 / 0.0243 / 0.0100 for the same
    recipe). On gemma-3-12b the sampler left rank 1 after 11 trials. Starting every search from the same grid
    makes the early frontier comparable across models and removes the lucky-start dependence."""
    norm = "none" if "none" in norm_choices else norm_choices[0]
    return [{"k": 1, "center_frac": c, "width_frac": w, "peak": pk, "kernel": "band", "kinds": "both",
             "norm_restore": norm, "subspace": sub, "broadcast": True}
            for sub in ("plain", "protected") for c in (0.50, 0.60) for w in (0.15, 0.25, 0.40) for pk in (0.8, 1.0)]


def frontier(study, levels=(0.02, 0.05, 0.10, 0.20, 0.50), upto: int | None = None) -> dict:
    """Lowest KL reached at or below each refusal level: {level: [kl, n_trials]} (None when no trial qualifies)."""
    pts = [(t.user_attrs.get("refusal_rate"), t.user_attrs.get("kl")) for t in study.trials
           if t.values is not None and (upto is None or t.number < upto)]
    pts = [(r, k) for r, k in pts if r is not None and k is not None]
    out = {}
    for lv in levels:
        ok = [k for r, k in pts if r <= lv]
        out[str(lv)] = [round(min(ok), 5), len(ok)] if ok else None
    return out


def run_study(model, tok, subspaces: dict, sets, trace, baseline: dict,
              originals: dict, n_trials: int = 40, study_name: str = "ablate",
              storage=None, seed: int = 0, device: str = "cuda",
              batch_size: int = 16, kl_batch_size: int = 8,
              lambda_kl: float = 1.0, lambda_cal: float = 2.0,
              judge=None, out_dir: Path | None = None,
              quant: str | None = None, linears: dict | None = None,
              edit_mode: str = "weights", objective_mode: str = "scalar",
              kl_target: float = 0.01, kl_scale: float = 1.0,
              n_startup: int | None = None, refusal_tol: float = 0.02,
              cal_tol: float = 0.05, penalty: float = 5.0,
              grid: list[dict] | None = None) -> "optuna.Study":
    """subspaces maps a variant name to a Subspace, normally protected and plain.

    quant names a preset from ablate.quant. When set, every trial is quantized
    before it is scored, so the search optimises the model that ships rather
    than the bfloat16 one it was edited in. This is the single change that
    matters most for a four bit artifact: quantization costs roughly an order
    of magnitude more KL than the edit does, so an optimum found at bfloat16 is
    an optimum for a different problem. The reference distribution stays the
    untouched bfloat16 model, because that is the behaviour being preserved.

    linears is a snapshot from quant.snapshot_linears, required with quant so
    each trial starts from unquantized weights. Quantizing a model that is
    already quantized compounds the error and quietly ruins the sweep.
    """
    if quant and quant not in PRESETS:
        raise ValueError(f"unknown quant preset {quant}")
    if quant and linears is None:
        raise ValueError("quant needs a linears snapshot to revert between trials")
    # edit_mode hook scores each trial through ablate.hookedit: the same function
    # as the weight edit with no weight written, so originals is not needed and
    # nothing has to be restored. It cannot be combined with quant, which rounds
    # the edited weights themselves.
    if edit_mode not in ("weights", "hook"):
        raise ValueError(edit_mode)
    if edit_mode == "hook" and quant:
        raise ValueError("edit_mode hook cannot score at a quant preset")
    import optuna
    from .modelio import has_block_writers, n_layers

    L = n_layers(model)
    # a mixture of experts block can only be hooked on its output, which is the
    # no restore edit; see ablate.hookedit
    norms = ("none",) if has_block_writers(model) else ("column", "none")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # objective_mode "pareto" (v3, 2026-11-11) is the two objective search Heretic uses, with our
    # protected behaviours on the behaviour axis. The scalar mode adds refusal rate and KL one to
    # one, so one residual refusal in 63 buys 0.016 KL and the search settles near KL 0.05; on the
    # bake-off that left us 9x above Heretic's KL at the same refusal rate. Here:
    #   behaviour = refusal_rate / baseline_refusal + lambda_cal * calibration drop
    #               + 0.5 * over refusal rise + 2 * coherence drop
    #   damage    = KL / kl_scale                     while KL >= kl_target
    #             = behaviour * kl_target / kl_scale  below it (stop exploring edits that do nothing)
    # and the pick is the lowest KL trial that clears the refusal and calibration tolerances.
    # objective_mode "kl_first" (v3 default candidate): single objective, constrained. Minimise KL;
    # refusal above refusal_tol, calibration drop above cal_tol, any over refusal rise or coherence
    # drop are penalised steeply (penalty 5 per unit, far above any KL we accept), so the sampler
    # spends its trials on the boundary "refusals just gone" instead of along the whole front.
    if objective_mode not in ("scalar", "pareto", "kl_first"):
        raise ValueError(objective_mode)
    if grid:
        grid = [g for g in grid if g.get("subspace") in subspaces]
    # with a grid the sampler starts modelling right after it: the grid IS the startup
    startup = (n_startup if n_startup is not None else
               (len(grid) if grid else max(10, int(0.3 * n_trials))))
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=startup)
    if objective_mode == "pareto":
        study = optuna.create_study(
            directions=["minimize", "minimize"], study_name=study_name, storage=storage,
            load_if_exists=storage is not None, sampler=sampler)
    else:
        study = optuna.create_study(
            direction="minimize", study_name=study_name, storage=storage,
            load_if_exists=storage is not None, sampler=sampler)
    base_ref = max(float(baseline.get("refusal_rate", 1.0)), 1e-6)

    def obj(trial):
        plan, k, variant = suggest_plan(trial, L, norm_choices=norms)
        if variant not in subspaces:
            variant = next(iter(subspaces))
        ss = subspaces[variant].with_rank(k)
        qstat = None
        if edit_mode == "hook":
            with HookEdit(model, ss, plan) as he:
                rec = he.record
                res = evaluate(model, tok, sets, trace=trace, judge=judge,
                               batch_size=batch_size, kl_batch_size=kl_batch_size,
                               device=device, with_samples=0)
        else:
            if linears is not None:
                restore_linears(model, linears)
            rec = apply_plan(model, ss, plan, originals=originals)
            try:
                if quant:
                    qstat = quantize_model_(model, **PRESETS[quant])
                res = evaluate(model, tok, sets, trace=trace, judge=judge,
                               batch_size=batch_size, kl_batch_size=kl_batch_size,
                               device=device, with_samples=0)
            finally:
                restore(model, originals)
                if linears is not None:
                    restore_linears(model, linears)
        sc = objective(baseline, res, lambda_kl=lambda_kl, lambda_cal=lambda_cal)
        for name, v in sc["terms"].items():
            trial.set_user_attr(name, float(v))
        trial.set_user_attr("n_edited", rec["n_edited"])
        trial.set_user_attr("variant", variant)
        trial.set_user_attr("scored_at", quant or "bf16")
        trial.set_user_attr("edit_mode", edit_mode)
        if qstat:
            trial.set_user_attr("rel_weight_rmse", qstat["rel_weight_rmse"])
        trial.set_user_attr("calibration", res.get("calibration", {}))
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"trial_{trial.number:04d}.json").write_text(
                json.dumps({"plan": asdict(plan), "k": k, "variant": variant,
                            "scored_at": quant or "bf16",
                            "score": sc, "result": res}, indent=2, default=str))
        trial.set_user_attr("scalar_value", float(sc["value"]))
        if objective_mode == "pareto":
            t = sc["terms"]
            behaviour = (t["refusal_rate"] / base_ref + lambda_cal * t["calibration_drop"]
                         + 0.5 * t["over_refusal_delta"] + 2.0 * t["coherence_drop"])
            damage = (t["kl"] / kl_scale if t["kl"] >= kl_target
                      else behaviour * kl_target / kl_scale)
            trial.set_user_attr("behaviour", float(behaviour))
            return damage, behaviour
        if objective_mode == "kl_first":
            t = sc["terms"]
            viol = (max(0.0, t["refusal_rate"] - refusal_tol) / base_ref
                    + max(0.0, t["calibration_drop"] - cal_tol)
                    + t["over_refusal_delta"] + t["coherence_drop"])
            trial.set_user_attr("violation", float(viol))
            return t["kl"] / kl_scale + penalty * viol
        return sc["value"]

    done = len([t for t in study.trials if t.values is not None])
    if done:
        print(f"resuming study: {done} trials already scored", flush=True)
    callbacks = []
    if grid:
        for g in grid:                                   # resumed studies skip cells already run or queued
            study.enqueue_trial(g, skip_if_exists=True)

        def report_grid(st, trial):
            if trial.number == len(grid) - 1:
                fr = frontier(st, upto=len(grid))
                st.set_user_attr("grid_frontier", fr)
                print(f"grid of {len(grid)} done; lowest KL at refusal <= level: {fr}", flush=True)
        callbacks.append(report_grid)
    study.optimize(obj, n_trials=max(0, n_trials - done), callbacks=callbacks)
    return study


def select_trial(study, refusal_tol: float = 0.02, cal_tol: float = 0.05):
    """Single objective: the best trial. Two objective (v3): the lowest KL trial whose refusal rate
    and calibration drop are inside the tolerances; if none qualifies, the lowest scalar value."""
    if len(study.directions) == 1 and not any("violation" in t.user_attrs for t in study.trials):
        return study.best_trial, "best scalar"
    done = [t for t in study.trials if t.values is not None and t.user_attrs.get("kl") is not None]
    ok = [t for t in done if t.user_attrs["refusal_rate"] <= refusal_tol
          and t.user_attrs.get("calibration_drop", 0.0) <= cal_tol
          and t.user_attrs.get("coherence_drop", 0.0) <= 0.02]
    if ok:
        return min(ok, key=lambda t: t.user_attrs["kl"]), f"lowest KL of {len(ok)} trials inside tolerance"
    return min(done, key=lambda t: t.user_attrs["scalar_value"]), "no trial inside tolerance, best scalar"


def best_plan(study, refusal_tol: float = 0.02, cal_tol: float = 0.05) -> dict:
    t, why = select_trial(study, refusal_tol, cal_tol)
    return {"value": t.user_attrs.get("scalar_value", t.values[0]), "selected_by": why,
            "trial": t.number, "params": t.params,
            "terms": {k: t.user_attrs.get(k) for k in
                      ("refusal_rate", "kl", "calibration_drop", "over_refusal_delta")},
            "variant": t.user_attrs.get("variant"),
            "n_edited": t.user_attrs.get("n_edited")}


def pareto(study, keys=("refusal_rate", "kl")) -> list[dict]:
    """Non dominated trials on the two headline axes, for the talk slide."""
    pts = [{"number": t.number, **{k: t.user_attrs.get(k) for k in keys},
            "params": t.params}
           for t in study.trials if t.values is not None
           and all(t.user_attrs.get(k) is not None for k in keys)]
    front = []
    for p in pts:
        if not any(q is not p and all(q[k] <= p[k] for k in keys)
                   and any(q[k] < p[k] for k in keys) for q in pts):
            front.append(p)
    return sorted(front, key=lambda x: x[keys[0]])
