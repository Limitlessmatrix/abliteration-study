"""Materialize edited weights from a finished run, on CPU, shard by shard.

  python scripts/bake_stream.py --base <dir with safetensors> --run runs/<name> --out <dir>

Reads search.json and subspace_<variant>.pt from the run directory, applies the
winning plan to the base checkpoint and writes a loadable model to --out. Use a
fast local disk for --out and copy from there; a slow network filesystem makes
big models crawl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ablate.streambake import bake, config_layers, plan_from_search, read_header, resolve_writers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True, help="run directory with search.json and subspace_*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args()

    from ablate.config import resolve_model_dir
    run, base = Path(a.run), Path(resolve_model_dir(a.base))   # local dir, or Hub id[@revision]
    p = json.loads((run / "search.json").read_text())["best"]["params"]
    names = [n for s in sorted(base.glob("*.safetensors")) for n in read_header(s)[0]]
    L = 1 + max(l for l, _ in resolve_writers(names, config_layers(base)).values())
    plan = plan_from_search(p, L)
    variant = p.get("subspace", "protected")
    sub = run / f"subspace_{variant}.pt"
    if not sub.exists():
        sub = run / "subspace_protected.pt"
    print(f"{L} layers, plan {plan}, subspace {sub.name}, k {p['k']}", flush=True)
    t0 = time.time()
    man = bake(base, Path(a.out), sub, plan, k=p["k"], device=a.device,
               hash_outputs=not a.no_hash, log=lambda m: print(m, flush=True))
    print(f"BAKE_DONE edited {man['n_edited']} skipped {man['skipped']} in {time.time() - t0:.0f} s -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
