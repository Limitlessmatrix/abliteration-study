"""Command line entry points.

  ablate build-data  --n-harmful 300        (or: python -m ablate.cli build-data ...)
  ablate extract     --model Qwen/Qwen2.5-1.5B-Instruct --out runs/qwen15
  ablate compare     --model Qwen/Qwen2.5-1.5B-Instruct --run runs/qwen15

The full v2 / v3.x pipeline (extract, layer scan, search, final scoring) is scripts/abliterate.py.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .config import data_dir

ROOT = Path(__file__).resolve().parent.parent
DATA = data_dir()   # ABLATE_DATA, default <repo>/data


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def cmd_build_data(a):
    from . import datasets as D
    s = D.build(DATA, harmful_source=a.harmful_source, n_harmful=a.n_harmful,
                n_corpus=a.n_corpus, seed=a.seed, twiner_model=a.twiner,
                skip_twins=a.skip_twins)
    print(json.dumps(s, indent=2))


def cmd_extract(a):
    from .modelio import load
    from .pipeline import ExtractConfig, extract, probe_separation, save_subspace
    from . import subspace as S

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"loading {a.model}", flush=True)
    model, tok = load(a.model, dtype=a.dtype, device=a.device)

    cfg = ExtractConfig(contrast=a.contrast, k=a.k, protect=not a.no_protect,
                        max_pairs=a.max_pairs, batch_size=a.batch_size,
                        split=a.split)
    art = extract(model, tok, DATA, cfg, device=a.device)
    probe = probe_separation(art)
    print("probe:", json.dumps(probe), flush=True)

    for name, v in art.variants.items():
        save_subspace(out / f"subspace_{name}.pt", v,
                      extra={"model": a.model, "config": asdict(cfg)},
                      condition=art.condition)
    save_subspace(out / "subspace.pt", art.subspace,
                  extra={"model": a.model, "config": asdict(cfg)},
                  condition=art.condition)
    _save(out / "extract.json", {
        "model": a.model, "config": asdict(cfg), "diag": art.diag,
        "probe": probe, "meta": art.subspace.meta,
        "best_layer": art.subspace.best_layer(),
        "separation": [round(float(x), 3) for x in art.subspace.separation],
        "rank": [int(x) for x in art.subspace.rank],
        "variance_ratio": [[round(float(y), 4) for y in row]
                           for row in art.subspace.variance_ratio],
        "protected_overlap": {k: [[round(float(y), 4) for y in row] for row in v]
                              for k, v in art.subspace.protected_overlap.items()},
        "report": S.report(art.subspace),
    })
    print(f"wrote {out/'subspace.pt'} and {out/'extract.json'}")


def cmd_compare(a):
    """Baseline against naive abliteration against the protected plan.

    This is the class demo: three models, one table, the damage visible in KL
    and in the calibration columns that KL does not capture.
    """
    from .apply import EditPlan, apply_plan, naive_plan, restore, snapshot
    from .capture import teacher_trace
    from .evalharness import evaluate, objective
    from .modelio import load, n_layers
    from .pipeline import build_eval_sets, load_subspace
    from .refusal import OllamaJudge

    run = Path(a.run)
    ss = load_subspace(run / "subspace.pt")
    print(f"loading {a.model}", flush=True)
    model, tok = load(a.model, dtype=a.dtype, device=a.device)
    sets = build_eval_sets(DATA, split=a.split, n=a.n_eval)
    judge = None
    if a.judge:
        judge = OllamaJudge(a.judge)
        if not judge.available():
            print(f"judge {a.judge} not available, falling back to markers", flush=True)
            judge = None

    print(f"caching reference trace on {len(sets.harmless)} harmless prompts", flush=True)
    trace = teacher_trace(model, tok, sets.harmless, max_new_tokens=a.kl_tokens,
                          top_k=a.top_k, batch_size=a.kl_batch_size, device=a.device)

    orig = snapshot(model)
    results = {}

    print("baseline", flush=True)
    results["baseline"] = evaluate(model, tok, sets, trace=trace, judge=judge,
                                   batch_size=a.batch_size, device=a.device)

    bl = ss.best_layer()
    L = n_layers(model)
    plans = {
        "naive": naive_plan(model, ss),
        "protected": EditPlan(center=float(bl), width=a.width, peak=a.peak,
                              kernel="gaussian", norm_restore=a.norm_restore),
    }
    for name, plan in plans.items():
        print(f"{name}  centre {plan.center:.1f} width {plan.width} peak {plan.peak} "
              f"kernel {plan.kernel} norm {plan.norm_restore}", flush=True)
        rec = apply_plan(model, ss, plan, originals=orig)
        results[name] = evaluate(model, tok, sets, trace=trace, judge=judge,
                                 batch_size=a.batch_size, device=a.device)
        results[name]["_edit"] = {"n_edited": rec["n_edited"], "plan": rec["plan"]}
        results[name]["_objective"] = objective(results["baseline"], results[name])
        restore(model, orig)

    _save(run / f"compare_{a.split}.json",
          {"model": a.model, "layers": L, "best_layer": bl, "results": results})
    print()
    print(_table(results))
    print(f"\nwrote {run/f'compare_{a.split}.json'}")


def cmd_search(a):
    """Stage 4. One extraction, many plans, scored on validation."""
    from .apply import snapshot
    from .capture import teacher_trace
    from .evalharness import evaluate
    from .modelio import load
    from .pipeline import build_eval_sets, load_subspace
    from .refusal import OllamaJudge
    from .search import best_plan, make_storage, pareto, run_study

    run = Path(a.run)
    variants = {}
    for name in ("protected", "plain"):
        p = run / f"subspace_{name}.pt"
        if p.exists():
            variants[name] = load_subspace(p)
    if not variants:
        variants["protected"] = load_subspace(run / "subspace.pt")

    print(f"loading {a.model}", flush=True)
    model, tok = load(a.model, dtype=a.dtype, device=a.device)
    sets = build_eval_sets(DATA, split=a.split, n=a.n_eval)
    judge = OllamaJudge(a.judge) if a.judge else None
    if judge is not None and not judge.available():
        judge = None

    print(f"caching reference trace on {len(sets.harmless)} harmless prompts", flush=True)
    trace = teacher_trace(model, tok, sets.harmless, max_new_tokens=a.kl_tokens,
                          top_k=a.top_k, batch_size=a.kl_batch_size, device=a.device)
    baseline = evaluate(model, tok, sets, trace=trace, judge=judge,
                        batch_size=a.batch_size, device=a.device, with_samples=0)
    print(f"baseline refusal {baseline.get('refusal_rate'):.3f}  "
          f"over-refusal {baseline.get('over_refusal_rate'):.3f}", flush=True)

    orig = snapshot(model)
    lin = None
    if a.quant:
        from .quant import snapshot_linears
        lin = snapshot_linears(model)
        print(f"scoring every trial at {a.quant}", flush=True)
    storage = make_storage(a.storage, path=str(run / "optuna.log"), url=a.storage_url)
    study = run_study(model, tok, variants, sets, trace, baseline, orig,
                      quant=a.quant or None, linears=lin,
                      n_trials=a.trials, study_name=a.study, storage=storage,
                      seed=a.seed, device=a.device, batch_size=a.batch_size,
                      kl_batch_size=a.kl_batch_size, lambda_kl=a.lambda_kl,
                      lambda_cal=a.lambda_cal, judge=judge,
                      out_dir=run / "trials")
    best = best_plan(study)
    front = pareto(study)
    _save(run / "search.json", {"model": a.model, "baseline": baseline,
                                "best": best, "pareto": front,
                                "n_trials": len(study.trials)})
    print("\nbest:", json.dumps(best, indent=2, default=str))
    print(f"pareto front: {len(front)} trials")
    print(f"wrote {run/'search.json'}")


def _table(results: dict) -> str:
    cats = ["false_premise", "unanswerable", "nonsense", "negative_constraint"]
    head = f"{'variant':<12}{'refusal':>9}{'over-ref':>10}{'KL':>9}" + \
           "".join(f"{c[:9]:>11}" for c in cats)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        cal = r.get("calibration", {})
        row = (f"{name:<12}{r.get('refusal_rate', float('nan')):>9.3f}"
               f"{r.get('over_refusal_rate', float('nan')):>10.3f}"
               f"{r.get('kl_mean', float('nan')):>9.4f}")
        for c in cats:
            v = cal.get(c, {}).get("behaviour_rate")
            row += f"{v:>11.3f}" if v is not None else f"{'-':>11}"
        lines.append(row)
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ablate", description="low damage abliteration")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data")
    b.add_argument("--harmful-source", default="advbench", choices=["advbench", "harmbench"])
    b.add_argument("--n-harmful", type=int, default=300)
    b.add_argument("--n-corpus", type=int, default=600)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--twiner", default=None)
    b.add_argument("--skip-twins", action="store_true")
    b.set_defaults(fn=cmd_build_data)

    e = sub.add_parser("extract")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--contrast", default="refuse_vs_comply",
                   choices=["refuse_vs_comply", "harmful_vs_harmless"])
    e.add_argument("--k", type=int, default=2)
    e.add_argument("--no-protect", action="store_true")
    e.add_argument("--max-pairs", type=int, default=256)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--split", default="train")
    e.add_argument("--dtype", default="bfloat16")
    e.add_argument("--device", default="cuda")
    e.set_defaults(fn=cmd_extract)

    c = sub.add_parser("compare")
    c.add_argument("--model", required=True)
    c.add_argument("--run", required=True)
    c.add_argument("--split", default="test")
    c.add_argument("--n-eval", type=int, default=64)
    c.add_argument("--kl-tokens", type=int, default=24)
    c.add_argument("--top-k", type=int, default=512)
    c.add_argument("--kl-batch-size", type=int, default=8)
    c.add_argument("--batch-size", type=int, default=16)
    c.add_argument("--width", type=float, default=3.0)
    c.add_argument("--peak", type=float, default=1.0)
    c.add_argument("--norm-restore", default="column", choices=["column", "row", "none"])
    c.add_argument("--judge", default=None)
    c.add_argument("--dtype", default="bfloat16")
    c.add_argument("--device", default="cuda")
    c.set_defaults(fn=cmd_compare)

    s = sub.add_parser("search")
    s.add_argument("--model", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--trials", type=int, default=40)
    s.add_argument("--study", default="ablate")
    s.add_argument("--storage", default="journal", choices=["journal", "postgres", "memory"])
    s.add_argument("--storage-url", default=None)
    s.add_argument("--split", default="val")
    s.add_argument("--n-eval", type=int, default=48)
    s.add_argument("--kl-tokens", type=int, default=24)
    s.add_argument("--top-k", type=int, default=512)
    s.add_argument("--kl-batch-size", type=int, default=8)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--lambda-kl", type=float, default=1.0)
    s.add_argument("--lambda-cal", type=float, default=2.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--judge", default=None)
    s.add_argument("--quant", default="",
                   help="score every trial at this precision, e.g. int4-g128")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--device", default="cuda")
    s.set_defaults(fn=cmd_search)

    a = p.parse_args(argv)
    t0 = time.time()
    a.fn(a)
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
