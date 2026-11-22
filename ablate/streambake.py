"""Bake an edit into a checkpoint one tensor at a time, with no model loaded.

The in memory route (apply_plan, save_pretrained, reload) needs the whole
model on GPUs plus a parked copy in host RAM, and then pushes 60 to 150 GB
through whatever the output path is. This does the same edit straight on the
safetensors shards: each shard is copied byte for byte, then only the residual
writer tensors are overwritten in place at their offsets. Peak memory is one
weight matrix, no GPU is needed, and every tensor the edit does not touch is
bit identical to the base by construction.

Inputs are what a run directory already holds: subspace_<variant>.pt and
search.json (or an explicit plan). Depends on torch only, so the machine that runs
it needs no transformers and no model code.

Writer resolution mirrors modelio.residual_writers on tensor names. It must
agree with it; scripts/selftest_big.py checks that on a real save_pretrained.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
from dataclasses import asdict
from pathlib import Path

import torch

ATTN_BLOCKS = ("self_attn", "attn", "attention", "linear_attn")
ATTN_OUT = ("o_proj", "out_proj", "dense", "wo")
MLP_OUT = ("down_proj", "fc2", "c_proj", "w2")
SKIP_TOWERS = ("vision", "visual", "audio")

_DT = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
_KEY = re.compile(
    r"^(?P<prefix>.*?)(?P<list>layers|h|blocks|block)\.(?P<i>\d+)\."
    r"(?P<blk>" + "|".join(ATTN_BLOCKS) + r"|mlp)\."
    r"(?P<sub>experts\.\d+\.|shared_experts\.)?(?P<lin>[A-Za-z0-9_]+)\.weight$")


def read_header(path: Path) -> tuple[dict, int]:
    """-> (tensor name -> {dtype, shape, data_offsets}, byte offset of the data buffer)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def resolve_writers(names: list[str], n_layers: int | None = None) -> dict[str, tuple[int, str]]:
    """tensor name -> (layer, kind) for the residual writers among names.

    One decoder stack only: the candidates must share a single prefix once the
    vision and audio towers are dropped. Per block the first name in ATTN_OUT
    or MLP_OUT order wins, which is what modelio._pick does. A mixture of
    experts layer contributes one mlp_out tensor per routed expert plus the
    shared expert.

    n_layers is num_hidden_layers from the config. Checkpoints with a multi
    token prediction head (GLM-5, DeepSeek V3) store it as one extra entry at
    the end of the layer list; it is not part of the decoder and is left alone.
    """
    cands: dict[str, dict] = {}
    for n in names:
        if any(t in n for t in SKIP_TOWERS):
            continue
        m = _KEY.match(n)
        if not m:
            continue
        kind = "mlp_out" if m["blk"] == "mlp" else "attn_out"
        order = MLP_OUT if kind == "mlp_out" else ATTN_OUT
        if m["lin"] not in order or (m["sub"] and kind != "mlp_out"):
            continue
        if n_layers is not None and int(m["i"]) >= n_layers:
            continue
        pre = m["prefix"] + m["list"]
        slot = (int(m["i"]), kind, m["sub"] or "")
        cur = cands.setdefault(pre, {}).get(slot)
        if cur is None or order.index(m["lin"]) < order.index(cur[1]):
            cands[pre][slot] = (n, m["lin"])
    if not cands:
        raise RuntimeError("no residual writer tensors found in the checkpoint")
    if len(cands) > 1:
        # the decoder is the stack with both kinds on every layer and the most layers
        full = {p: s for p, s in cands.items()
                if {k for _, k, _ in s} == {"attn_out", "mlp_out"}}
        best = max(full or cands, key=lambda p: len(cands[p]))
        rest = [p for p in cands if p != best and len(cands[p]) == len(cands[best])]
        if rest:
            raise RuntimeError(f"ambiguous decoder stacks: {best} vs {rest}")
        cands = {best: cands[best]}
    (_, slots), = cands.items()
    return {name: slot[:2] for slot, (name, _) in slots.items()}


def config_layers(base: Path) -> int | None:
    """num_hidden_layers of the language model, if the checkpoint has a config."""
    p = Path(base) / "config.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return (c.get("text_config") or c).get("num_hidden_layers")


class _Sub:
    """The part of ablate.subspace.Subspace the edit needs, read from the .pt."""

    def __init__(self, path: Path, k: int | None):
        d = torch.load(path, map_location="cpu", weights_only=True)
        self.basis, self.rank, self.k = d["basis"], d["rank"], k

    def layer(self, l: int) -> torch.Tensor:
        r = int(self.rank[l])
        if self.k is not None:
            r = min(r, int(self.k))
        return self.basis[l][:r]


def plan_from_search(params: dict, n_layers: int):
    """Same mapping as scripts/abliterate.py uses after Stage 4."""
    from .apply import EditPlan
    center = params["center_frac"] * (n_layers - 1)
    width = (params["width_frac"] * (n_layers - 1) if "width_frac" in params
             else params.get("width", 3.5))
    kinds = {"both": ("attn_out", "mlp_out"), "mlp_only": ("mlp_out",),
             "attn_only": ("attn_out",)}[params["kinds"]]
    return EditPlan(center=center, width=width, peak=params["peak"], kernel=params["kernel"],
                    kinds=kinds, norm_restore=params["norm_restore"],
                    source_layer=int(round(center)) if params.get("broadcast") else None)


def _sha256(path: Path, buf: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def bake(base: Path, out: Path, subspace_pt: Path, plan, k: int | None = None,
         device: str = "cpu", hash_outputs: bool = True, log=print) -> dict:
    """Write base with the plan applied to out. Returns the manifest (also saved)."""
    from .apply import _project_columns

    base, out = Path(base), Path(out)
    shards = sorted(base.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors in {base}")
    if out.resolve() == base.resolve():
        raise RuntimeError("out must differ from base; the base is never modified")
    out.mkdir(parents=True, exist_ok=True)

    headers = {s: read_header(s) for s in shards}
    names = [n for h, _ in headers.values() for n in h]
    writers = resolve_writers(names, config_layers(base))
    n_layers = 1 + max(l for l, _ in writers.values())
    ss = _Sub(subspace_pt, k)
    if plan.edit_embedding:
        raise NotImplementedError("stream bake does not edit the embedding")

    edited, skipped = [], 0
    for s in shards:
        hdr, off0 = headers[s]
        dst = out / s.name
        todo = [n for n in hdr if n in writers and writers[n][1] in plan.kinds]
        tmp = dst.with_suffix(".part")
        shutil.copyfile(s, tmp)
        with open(tmp, "r+b") as f:
            for n in todo:
                layer, kind = writers[n]
                a = plan.alpha(layer)
                if a < plan.min_alpha:
                    skipped += 1
                    continue
                src = plan.source_layer if plan.source_layer is not None else layer
                src = max(0, min(int(src), ss.basis.shape[0] - 1))
                Q = ss.layer(src)
                if Q.shape[0] == 0:
                    skipped += 1
                    continue
                meta = hdr[n]
                if meta["dtype"] not in _DT:
                    raise RuntimeError(f"{n}: dtype {meta['dtype']} is not a float weight")
                b0, b1 = meta["data_offsets"]
                f.seek(off0 + b0)
                raw = bytearray(f.read(b1 - b0))
                W = torch.frombuffer(raw, dtype=_DT[meta["dtype"]]).reshape(meta["shape"])
                if W.shape[0] != Q.shape[1]:
                    raise RuntimeError(f"{n}: rows {W.shape[0]} != hidden {Q.shape[1]}")
                W2 = _project_columns(W.to(device), Q, a, plan.norm_restore).cpu().contiguous()
                if not torch.isfinite(W2.float()).all():
                    raise RuntimeError(f"{n}: non finite values after the edit")
                f.seek(off0 + b0)
                f.write(W2.view(torch.uint8).numpy().tobytes())
                edited.append({"tensor": n, "file": s.name, "layer": layer, "kind": kind,
                               "alpha": round(a, 4), "rank": int(Q.shape[0]), "source": src})
                del W, W2, raw
        tmp.replace(dst)
        log(f"{s.name}: {len(todo)} writer tensors, {dst.stat().st_size / 1e9:.2f} GB")

    want = {n for n, (_, kd) in writers.items() if kd in plan.kinds}
    if len(edited) + skipped != len(want):
        raise RuntimeError(f"edited {len(edited)} + skipped {skipped} != {len(want)} writers")

    for p in base.iterdir():      # config, tokenizer, template, index: everything but weights
        if p.is_file() and p.suffix not in (".safetensors", ".part", ".bin", ".gguf"):
            shutil.copyfile(p, out / p.name)

    man = {"base": str(base), "subspace": str(subspace_pt), "k": k, "plan": asdict(plan),
           "n_layers": n_layers, "n_edited": len(edited), "skipped": skipped, "edited": edited}
    if hash_outputs:
        man["sha256"] = {s.name: _sha256(out / s.name) for s in shards}
    (out / "bake_manifest.json").write_text(json.dumps(man, indent=1))
    return man
