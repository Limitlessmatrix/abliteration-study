"""Extract and choose an edit for a model that does not fit on the GPUs.

  python scripts/abliterate_stream.py --model <dir> --out runs/<name> \
      --gpu-gib 72 --cpu-gib 200 --offload /path/to/fast/local/disk

The model is held by accelerate across the GPUs, host RAM and disk, so every
forward pass re-reads whatever is offloaded. Nothing here generates: Stage 2 is
one capture pass per prompt set, and Stage 4 is ablate.streamsearch, one pass
per candidate. No weights are written. The run directory ends up with the same
subspace_*.pt and search.json a normal run has, and scripts/bake_stream.py turns
those into a checkpoint on a CPU node.

Differences from scripts/abliterate.py that the card records:
  no behaviour labels (they need generation), so the contrast is always bulk
  no refusal rate, coherence or response KL; only first token metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from ablate import calibration as CAL
from ablate import datasets as D
from ablate import subspace as S
from ablate.capture import capture_residuals
from ablate.modelio import has_block_writers, hidden_size, load, n_layers, residual_writers
from ablate.pipeline import load_subspace, save_subspace
from ablate.streamsearch import candidates, score_candidates

from ablate.config import data_dir
DATA = data_dir()   # ABLATE_DATA, default <repo>/data


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64,
                    help="prompts per forward pass; each pass re-reads the offloaded weights")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-gib", type=int, default=0, help="per GPU budget for weights; 0 lets accelerate decide")
    ap.add_argument("--cpu-gib", type=int, default=0)
    ap.add_argument("--offload", default="")
    ap.add_argument("--kl-budget", type=float, default=0.1)
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--peaks", default="1.0")
    ap.add_argument("--kernels", default="uniform,band")
    ap.add_argument("--variants", default="plain")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mm = None
    if a.gpu_gib:
        mm = {i: f"{a.gpu_gib}GiB" for i in range(torch.cuda.device_count())}
        if a.cpu_gib:
            mm["cpu"] = f"{a.cpu_gib}GiB"
    model, tok = load(a.model, dtype=a.dtype, device=a.device, max_memory=mm,
                      offload_folder=a.offload or None)
    L = n_layers(model)
    ws = residual_writers(model)
    placed = getattr(model, "hf_device_map", None)
    where = {}
    for v in (placed or {}).values():
        where[str(v)] = where.get(str(v), 0) + 1
    log(f"{L} layers, hidden {hidden_size(model)}, {len(ws)} writers "
        f"({sum(w.block for w in ws)} expert blocks), placement {where or a.device}")

    cap = lambda ps: capture_residuals(model, tok, ps, batch_size=a.batch, device=a.device,
                                       positions="post_instruction", clip=a.clip)
    if (out / "extract.done").exists():
        log("stage 2 cached")
        variants = {n: load_subspace(out / f"subspace_{n}.pt") for n in ("protected", "plain")
                    if (out / f"subspace_{n}.pt").exists()}
    else:
        tr = D.load(DATA, split="train")
        pairs, cal_all = tr["pairs"][: a.pairs], tr["calibration"]
        bulk = [c["prompt"] for c in tr["corpus"]][: len(pairs)]
        A = {}
        for name, ps in (("harmful", [p["harmful"] for p in pairs]), ("bulk_harmless", bulk),
                         ("cal_probe", [c["probe"] for c in cal_all]),
                         ("cal_control", [c["control"] for c in cal_all])):
            t = time.time()
            A[name] = cap(ps)
            log(f"captured {name}: {tuple(A[name].shape)} in {time.time() - t:.0f} s")
        protected = {}
        for c in CAL.CATEGORIES:
            idx = [i for i, x in enumerate(cal_all) if x["category"] == c]
            if len(idx) >= 4:
                protected[c] = S.behaviour_directions(A["cal_probe"][idx], A["cal_control"][idx])
        hm = A["bulk_harmless"].double().mean(0).float()
        variants = {n: S.build(A["harmful"], A["bulk_harmless"], k=a.k, paired=False,
                               harmless_mean=hm if pr else None, protected=protected, protect=pr)
                    for n, pr in (("protected", True), ("plain", False))}
        for n, v in variants.items():
            save_subspace(out / f"subspace_{n}.pt", v, extra={"model": a.model})
        meta = {"report": S.report(variants["protected"]), "contrast": "bulk", "clip": a.clip,
                "k": a.k, "n_pairs": len(pairs), "labels": "none, stream mode does not generate"}
        (out / "extract.json").write_text(json.dumps(meta, indent=2))
        (out / "extract.done").touch()
        log(meta["report"])

    if (out / "search.done").exists():
        log("stage 4 cached")
    else:
        val = D.load(DATA, split="val")
        harmful = [p["harmful"] for p in val["pairs"]][: a.n_eval]
        harmless = [c["prompt"] for c in val["corpus"]][: a.n_eval]
        bl = variants["plain"].best_layer()
        cands = candidates(L, bl, ks=[int(x) for x in a.ks.split(",")],
                           peaks=[float(x) for x in a.peaks.split(",")],
                           kernels=a.kernels.split(","), variants=a.variants.split(","),
                           norms=("none",) if has_block_writers(model) else ("column",))
        log(f"best layer {bl} by separation; scoring {len(cands)} candidates, one pass each")
        res = score_candidates(model, tok, variants, harmful, harmless, cands,
                               batch_size=a.batch, kl_budget=a.kl_budget,
                               out_dir=out / "cands", log=log)
        res["best_layer"] = bl
        (out / "search.json").write_text(json.dumps(res, indent=2))
        (out / "search.done").touch()
        log(f"best {res['best']['params']}  {res['best']['terms']}")
    log(f"STREAM_DONE {time.time() - t0:.0f} s -> {out}; bake with scripts/bake_stream.py")


if __name__ == "__main__":
    main()
