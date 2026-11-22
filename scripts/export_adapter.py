"""Write a finished run's edit as a LoRA adapter (PEFT layout), on CPU, with no model loaded.

  python scripts/export_adapter.py --base <dir with safetensors> --run runs/<run> --out <dir>
         [--mode exact|svd] [--rank 16] [--base-name org/model@rev] [--verify <baked checkpoint dir>]
  python scripts/export_adapter.py --selftest

Same inputs as bake_stream.py (search.json and subspace_<variant>.pt from the run directory). --verify merges the
adapter into the base in memory and compares it with a baked checkpoint of the same run (exit 1 on a mismatch).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ablate import adapter as AD  # noqa: E402
from ablate.streambake import bake, config_layers, plan_from_search, read_header, resolve_writers  # noqa: E402


def selftest() -> int:
    from ablate.apply import EditPlan
    torch.manual_seed(0)
    hidden, d_in, L, k = 64, 96, 6, 3
    checks = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = td / "base"
        base.mkdir()
        t = {}
        for i in range(L):
            t[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
            t[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, d_in)
            t[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(d_in, hidden)
        t["model.embed_tokens.weight"] = torch.randn(100, hidden)
        AD.save_safetensors({n: v for n, v in t.items()}, base / "model.safetensors")
        # the bake reads the dtype from the header: rewrite as bf16 to test the rounding path too
        (base / "config.json").write_text(json.dumps({"num_hidden_layers": L}))
        basis = torch.linalg.qr(torch.randn(L, hidden, k))[0].transpose(1, 2).contiguous()   # [L, k, hidden]
        torch.save({"basis": basis, "rank": torch.full((L,), k)}, td / "sub.pt")
        for restore, mode in (("none", "exact"), ("column", "svd")):
            plan = EditPlan(center=2.5, width=2.0, peak=0.97, kernel="band", kinds=("attn_out", "mlp_out"),
                            norm_restore=restore, source_layer=None)
            baked, ad = td / f"baked-{restore}", td / f"ad-{restore}"
            bm = bake(base, baked, td / "sub.pt", plan, k=k, hash_outputs=False, log=lambda m: None)
            man = AD.export(base, ad, td / "sub.pt", plan, k=k, mode=mode, r=hidden, log=lambda m: None)
            rep = AD.verify_against_bake(base, baked, ad, log=lambda m: None)
            checks += [(f"{restore}: same tensors as the bake", man["n_edited"] == bm["n_edited"] > 0),
                       (f"{restore}: untouched tensors identical", rep["untouched_differ"] == 0 and not rep["missing"]),
                       (f"{restore}: merged equals baked", rep["max_abs"] < 2e-5)]
            if restore == "none":
                cfg = json.loads((ad / "adapter_config.json").read_text())
                tens = AD.load_safetensors(ad / "adapter_model.safetensors")
                checks += [("exact: rank is k", man["rank"] == k and man["exact"] and man["max_residual"] == 0.0),
                           ("exact: peft keys", all(n.startswith("base_model.model.model.layers.") for n in tens)),
                           ("exact: shapes", all(v.shape[0] == k if "lora_A" in n else v.shape[1] == k for n, v in tens.items())),
                           ("exact: targets listed", len(cfg["target_modules"]) == man["n_edited"] and cfg["lora_alpha"] == cfg["r"])]
            else:
                low = AD.export(base, td / "ad-low", td / "sub.pt", plan, k=k, mode="svd", r=k, log=lambda m: None)
                checks += [("column: full-rank fit is exact", man["max_residual"] < 1e-4),
                           ("column: rank k is NOT exact (restore is not low rank)", low["max_residual"] > 1e-3)]
                try:
                    AD.export(base, td / "ad-x", td / "sub.pt", plan, k=k, mode="exact", log=lambda m: None)
                    checks.append(("column: exact mode refuses", False))
                except ValueError:
                    checks.append(("column: exact mode refuses", True))
    bad = [n for n, ok in checks if not ok]
    for n in bad:
        print("SELFTEST FAIL:", n)
    print(f"export_adapter selftest: {len(checks) - len(bad)}/{len(checks)} pass")
    return 1 if bad else 0


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="exact", choices=["exact", "svd"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--base-name", default="")
    ap.add_argument("--verify", default="")
    a = ap.parse_args()
    from ablate.config import resolve_model_dir
    run, base = Path(a.run), Path(resolve_model_dir(a.base))   # local dir, or Hub id[@revision]
    p = json.loads((run / "search.json").read_text())["best"]["params"]
    names = [n for s in sorted(base.glob("*.safetensors")) for n in read_header(s)[0]]
    L = 1 + max(l for l, _ in resolve_writers(names, config_layers(base)).values())
    plan = plan_from_search(p, L)
    sub = run / f"subspace_{p.get('subspace', 'protected')}.pt"
    if not sub.exists():
        sub = run / "subspace_protected.pt"
    print(f"{L} layers, plan {plan}, subspace {sub.name}, k {p['k']}", flush=True)
    man = AD.export(base, Path(a.out), sub, plan, k=p["k"], mode=a.mode, r=a.rank, base_name=a.base_name,
                    log=lambda m: print(m, flush=True))
    print(f"ADAPTER_DONE edited {man['n_edited']} rank {man['rank']} bytes {man['bytes']}", flush=True)
    if a.verify:
        rep = AD.verify_against_bake(base, Path(a.verify), Path(a.out), log=lambda m: print(m, flush=True))
        (Path(a.out) / "verify.json").write_text(json.dumps(rep, indent=1))
        sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
