"""Compare two safetensors checkpoints tensor by tensor, without loading a model.

  python scripts/compare_ckpt.py --a <dir> --b <dir> [--base <dir>] [--json out.json]

Shard layout may differ; tensors are matched by name. With --base, tensors are
split into those the edit changed (b differs from base) and those it did not,
which is what tells an edit that agrees from two that both did nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from ablate.streambake import _DT, read_header


class Ckpt:
    def __init__(self, d):
        self.where = {}
        for s in sorted(Path(d).glob("*.safetensors")):
            hdr, off = read_header(s)
            for n, m in hdr.items():
                self.where[n] = (s, off, m)

    def raw(self, n) -> bytes:
        s, off, m = self.where[n]
        b0, b1 = m["data_offsets"]
        with open(s, "rb") as f:
            f.seek(off + b0)
            return f.read(b1 - b0)

    def tensor(self, n, raw=None):
        _, _, m = self.where[n]
        return torch.frombuffer(bytearray(raw or self.raw(n)), dtype=_DT[m["dtype"]]).reshape(m["shape"]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--tol", type=float, default=2e-3, help="pass if every relative Frobenius difference is below this")
    a = ap.parse_args()

    A, B = Ckpt(a.a), Ckpt(a.b)
    base = Ckpt(a.base) if a.base else None
    only_a, only_b = sorted(set(A.where) - set(B.where)), sorted(set(B.where) - set(A.where))
    same = diff = edited = 0
    worst_rel = worst_abs = 0.0
    frac_elems = []
    unexpected = []          # differs between a and b although the edit left it alone
    for n in sorted(set(A.where) & set(B.where)):
        ra, rb = A.raw(n), B.raw(n)
        was_edited = base is not None and n in base.where and base.raw(n) != rb
        edited += was_edited
        if ra == rb:
            same += 1
            continue
        diff += 1
        if A.where[n][2]["dtype"] not in _DT or A.where[n][2]["shape"] != B.where[n][2]["shape"]:
            unexpected.append(n)
            continue
        ta, tb = A.tensor(n, ra), B.tensor(n, rb)
        d = ta - tb
        rel = float(d.norm() / tb.norm().clamp(min=1e-12))
        worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, float(d.abs().max()))
        frac_elems.append(float((d != 0).float().mean()))
        if base is not None and not was_edited:
            unexpected.append(n)
    ok = not only_a and not only_b and not unexpected and worst_rel < a.tol and (base is None or edited > 0)
    res = {"a": a.a, "b": a.b, "tensors": len(set(A.where) & set(B.where)), "bit_identical": same,
           "differing": diff, "edited_vs_base": edited if base else None,
           "worst_rel_fro": worst_rel, "worst_abs": worst_abs,
           "mean_frac_elements_differing": sum(frac_elems) / len(frac_elems) if frac_elems else 0.0,
           "only_in_a": only_a[:8], "only_in_b": only_b[:8], "unexpected": unexpected[:8],
           "n_unexpected": len(unexpected), "verdict": "PASS" if ok else "FAIL"}
    print(json.dumps(res, indent=1), flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(f"COMPARE_{res['verdict']}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
