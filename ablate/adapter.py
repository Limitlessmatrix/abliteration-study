"""Export a finished run's edit as a LoRA adapter instead of an edited checkpoint.

The edit on a residual writer W [hidden, d_in] with orthonormal rows Q [k, hidden] is
    W' = W - alpha * Q^T Q W                       (norm_restore = none)
which is W + B A with B = -alpha * Q^T  [hidden, k] and A = Q W  [k, d_in]: a rank-k LoRA, exact.
A is computed from the base weights, so the adapter belongs to ONE base checkpoint (any quantisation of it).

With norm_restore = column the edit is W' = (W - alpha Q^T Q W) S with S a diagonal column rescale. That is not low
rank (W (S - I) has full rank); it is what DoRA calls the magnitude vector. `mode="svd"` then fits the true delta
with a truncated SVD of rank `r` and reports the residual per tensor; `mode="exact"` refuses.

Output is the PEFT layout (adapter_model.safetensors + adapter_config.json), which transformers / peft load and
llama.cpp's convert_lora_to_gguf.py converts. No transformers or peft needed here: torch and the checkpoint shards.

Inputs are the same a bake uses (streambake): the base checkpoint, subspace_<variant>.pt and the plan.
"""
from __future__ import annotations

import json
import struct
from dataclasses import asdict
from pathlib import Path

import torch

from .streambake import _DT, _Sub, config_layers, read_header, resolve_writers


def _read_tensor(f, off0: int, meta: dict) -> torch.Tensor:
    b0, b1 = meta["data_offsets"]
    f.seek(off0 + b0)
    raw = bytearray(f.read(b1 - b0))
    return torch.frombuffer(raw, dtype=_DT[meta["dtype"]]).reshape(meta["shape"]).clone()


def save_safetensors(tensors: dict, path: Path, metadata: dict | None = None) -> None:
    """Minimal writer (float32 tensors), so the export needs no safetensors package."""
    hdr, off, blobs = {}, 0, []
    for name in sorted(tensors):
        t = tensors[name].to(torch.float32).contiguous()
        b = t.numpy().tobytes()
        hdr[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    if metadata:
        hdr["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def load_safetensors(path: Path) -> dict:
    hdr, off0 = read_header(path)
    out = {}
    with open(path, "rb") as f:
        for n, meta in hdr.items():
            out[n] = _read_tensor(f, off0, meta)
    return out


@torch.no_grad()
def factors(W: torch.Tensor, Q: torch.Tensor, alpha: float, norm_restore: str = "none",
            mode: str = "exact", r: int = 16) -> tuple[torch.Tensor, torch.Tensor, float]:
    """-> (B [hidden, rank], A [rank, d_in], relative residual of B A against the true delta)."""
    Wf, Qf = W.float(), Q.float()
    if norm_restore == "none":
        return (-alpha * Qf.T).contiguous(), (Qf @ Wf).contiguous(), 0.0
    if mode != "svd":
        raise ValueError(f"norm_restore={norm_restore} is not a low-rank edit; use mode='svd'")
    from .apply import _project_columns
    delta = _project_columns(Wf, Qf, alpha, norm_restore) - Wf
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    r = min(r, S.shape[0])
    B, A = (U[:, :r] * S[:r]).contiguous(), Vh[:r].contiguous()
    res = float((delta - B @ A).norm() / delta.norm().clamp(min=1e-30))
    return B, A, res


@torch.no_grad()
def export(base: Path, out: Path, subspace_pt: Path, plan, k: int | None = None, mode: str = "exact",
           r: int = 16, base_name: str = "", log=print) -> dict:
    """Write the adapter for `plan` to out/. Returns the manifest (also saved as adapter_manifest.json)."""
    base, out = Path(base), Path(out)
    shards = sorted(base.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors in {base}")
    out.mkdir(parents=True, exist_ok=True)
    headers = {s: read_header(s) for s in shards}
    names = [n for h, _ in headers.values() for n in h]
    writers = resolve_writers(names, config_layers(base))
    ss = _Sub(subspace_pt, k)
    if plan.edit_embedding:
        raise NotImplementedError("adapter export does not cover an embedding edit")
    pairs, edited, skipped = {}, [], 0
    for s in shards:
        hdr, off0 = headers[s]
        with open(s, "rb") as f:
            for n in hdr:
                if n not in writers or writers[n][1] not in plan.kinds:
                    continue
                layer, kind = writers[n]
                a = plan.alpha(layer)
                src = plan.source_layer if plan.source_layer is not None else layer
                src = max(0, min(int(src), ss.basis.shape[0] - 1))
                Q = ss.layer(src)
                if a < plan.min_alpha or Q.shape[0] == 0:
                    skipped += 1
                    continue
                W = _read_tensor(f, off0, hdr[n])
                if W.shape[0] != Q.shape[1]:
                    raise RuntimeError(f"{n}: rows {W.shape[0]} != hidden {Q.shape[1]}")
                B, A, res = factors(W, Q, a, plan.norm_restore, mode, r)
                if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
                    raise RuntimeError(f"{n}: non finite adapter values")
                pairs[n] = (B, A)
                edited.append({"tensor": n, "layer": layer, "kind": kind, "alpha": round(a, 4),
                               "rank": int(A.shape[0]), "residual": round(res, 6),
                               "delta_over_w": round(float((B @ A).norm() / W.float().norm()), 6)})
    if not pairs:
        raise RuntimeError("the plan edits no tensor")
    rank = max(A.shape[0] for _, A in pairs.values())
    tensors, modules = {}, []
    for n, (B, A) in pairs.items():
        mod = n[: -len(".weight")]
        modules.append(mod)
        if A.shape[0] < rank:                       # PEFT wants one rank: pad with zero rows / columns
            A = torch.cat([A, A.new_zeros(rank - A.shape[0], A.shape[1])], 0)
            B = torch.cat([B, B.new_zeros(B.shape[0], rank - B.shape[1])], 1)
        tensors[f"base_model.model.{mod}.lora_A.weight"] = A
        tensors[f"base_model.model.{mod}.lora_B.weight"] = B
    save_safetensors(tensors, out / "adapter_model.safetensors", {"format": "pt"})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_name or str(base),
           "r": rank, "lora_alpha": rank, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
           "inference_mode": True, "init_lora_weights": True, "target_modules": sorted(modules),
           "modules_to_save": None, "use_dora": False, "use_rslora": False}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    size = (out / "adapter_model.safetensors").stat().st_size
    man = {"base": base_name or str(base), "subspace": str(subspace_pt), "k": k, "plan": asdict(plan), "mode": mode,
           "rank": rank, "n_edited": len(edited), "skipped": skipped, "bytes": size,
           "exact": plan.norm_restore == "none",
           "max_residual": max(e["residual"] for e in edited), "edited": edited}
    (out / "adapter_manifest.json").write_text(json.dumps(man, indent=1))
    log(f"adapter: {len(edited)} tensors, rank {rank}, {size / 1e6:.1f} MB, exact {man['exact']}, "
        f"max residual {man['max_residual']}")
    return man


@torch.no_grad()
def verify_against_bake(base: Path, baked: Path, adapter_dir: Path, log=print) -> dict:
    """Merge the adapter into the base in memory, tensor by tensor, and compare with a baked checkpoint.

    The bake stores W' rounded to the checkpoint dtype, so equality is checked after the same rounding: `mismatch`
    counts elements that differ after rounding, `worst_ulp` is the largest gap in units of the dtype's spacing.
    Also checks that every tensor the adapter does not touch is bit identical in the two checkpoints.
    """
    ad = load_safetensors(Path(adapter_dir) / "adapter_model.safetensors")
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    touched = {m + ".weight" for m in cfg["target_modules"]}
    rep = {"tensors": 0, "elements": 0, "mismatch": 0, "worst_ulp": 0.0, "max_abs": 0.0,
           "untouched_differ": 0, "missing": []}
    for s in sorted(Path(base).glob("*.safetensors")):
        hb, ob = read_header(s)
        hk, ok = read_header(Path(baked) / s.name)
        with open(s, "rb") as fb, open(Path(baked) / s.name, "rb") as fk:
            for n, meta in hb.items():
                if n not in hk:
                    rep["missing"].append(n)
                    continue
                if n not in touched:
                    b0, b1 = meta["data_offsets"]
                    k0, k1 = hk[n]["data_offsets"]
                    fb.seek(ob + b0)
                    fk.seek(ok + k0)
                    if fb.read(b1 - b0) != fk.read(k1 - k0):
                        rep["untouched_differ"] += 1
                    continue
                W = _read_tensor(fb, ob, meta)
                Wk = _read_tensor(fk, ok, hk[n])
                mod = n[: -len(".weight")]
                A = ad[f"base_model.model.{mod}.lora_A.weight"]
                B = ad[f"base_model.model.{mod}.lora_B.weight"]
                M = (W.float() + scale * (B @ A)).to(W.dtype)
                d = (M.float() - Wk.float()).abs()
                # spacing of the stored dtype at |x| lies in [eps*|x|/2, eps*|x|]; the lower bound is used, so one
                # rounding step shows as at most 2 here
                ulp = Wk.float().abs() * (torch.finfo(Wk.dtype).eps / 2)
                # where the edit nearly cancels the weight the result is tiny and float32 error scales with |W|,
                # not with the result: allow a few float32 steps of the unedited weight on top
                ulp = ulp + W.float().abs() * (4 * torch.finfo(torch.float32).eps)
                rep["tensors"] += 1
                rep["elements"] += d.numel()
                rep["mismatch"] += int((d > 0).sum())
                rep["worst_ulp"] = max(rep["worst_ulp"], float((d / ulp.clamp(min=1e-45)).max()))
                rep["max_abs"] = max(rep["max_abs"], float(d.max()))
    rep["mismatch_frac"] = rep["mismatch"] / max(1, rep["elements"])
    rep["ok"] = (rep["tensors"] == len(touched) and not rep["missing"] and rep["untouched_differ"] == 0
                 and rep["worst_ulp"] <= 2.0)
    log(f"verify: {rep['tensors']} tensors, {rep['mismatch']} of {rep['elements']} elements differ after rounding "
        f"({rep['mismatch_frac']:.2e}), worst {rep['worst_ulp']:.2f} ulp, max abs {rep['max_abs']:.2e}, untouched differing {rep['untouched_differ']} "
        f"-> {'PASS' if rep['ok'] else 'FAIL'}")
    return rep
