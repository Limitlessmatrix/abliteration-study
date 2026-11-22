"""Quantization: measuring whether an edit survives it, and training so it does.

Why this is not an afterthought. Abliteration is a small low rank perturbation
to a handful of matrices. Four bit quantization is per weight noise of
comparable magnitude applied to all of them. So the two interact, and the
direction of the interaction is not obvious in advance: quantization error can
partially restore the refusal direction that was projected out, or it can
compound the damage the edit already did. A refusal rate measured at bfloat16
is a number about a model nobody is going to run.

Two things live here. A quantize dequantize that is exact enough to measure
with, and a parametrization that puts the same op in the forward pass with a
straight through estimator so a healing LoRA can learn to compensate for it.

Fidelity note. Group wise affine int4 is the scheme behind GPTQ, AWQ and
bitsandbytes NF4, and it is close to GGUF Q4_K. It is not identical: llama.cpp
k-quants use per superblock scales with mixed bit allocation across a block.
Treat these numbers as the quantization sensitivity of the edit, and confirm
the final artifact through the real GGUF conversion before shipping it.
"""
from __future__ import annotations

import torch
from torch import nn

SKIP = ("lm_head", "embed_tokens", "embed_out")


def quantize_dequantize(W: torch.Tensor, bits: int = 4, group_size: int = 128,
                        symmetric: bool = False) -> torch.Tensor:
    """Round trip W through a bits-wide grid, grouped along the input dim.

    Grouping along the input dimension is what GPTQ, AWQ and the GGUF k-quants
    all do: consecutive input channels share a scale.
    """
    if bits >= 16:
        return W
    orig_dtype, orig_shape = W.dtype, W.shape
    X = W.float()
    n_in = X.shape[-1]
    gs = n_in if group_size in (0, -1) or group_size > n_in else group_size
    pad = (-n_in) % gs
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    X = X.reshape(-1, X.shape[-1] // gs, gs)

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = X.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        q = (X / scale).round().clamp(-qmax - 1, qmax)
        Y = q * scale
    else:
        qmax = 2 ** bits - 1
        lo = X.amin(-1, keepdim=True)
        hi = X.amax(-1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp(min=1e-8)
        zero = (-lo / scale).round()
        q = (X / scale + zero).round().clamp(0, qmax)
        Y = (q - zero) * scale

    Y = Y.reshape(orig_shape[0], -1)
    if pad:
        Y = Y[:, :n_in]
    return Y.reshape(orig_shape).to(orig_dtype)


def _targets(model, skip=SKIP):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(s in name for s in skip):
            yield name, mod


@torch.no_grad()
def quantize_model_(model, bits: int = 4, group_size: int = 128,
                    symmetric: bool = False, skip=SKIP) -> dict:
    """Quantize dequantize every decoder Linear in place.

    Embeddings and the output head are left alone, which is what llama.cpp does
    by default and what every serving stack does in practice.
    """
    n, num, den = 0, 0.0, 0.0
    for _, mod in _targets(model, skip):
        W = mod.weight.data
        Q = quantize_dequantize(W, bits, group_size, symmetric)
        num += float((Q.float() - W.float()).pow(2).sum())
        den += float(W.float().pow(2).sum())
        mod.weight.data.copy_(Q)
        n += 1
    return {"bits": bits, "group_size": group_size, "symmetric": symmetric,
            "tensors": n, "rel_weight_rmse": (num / max(den, 1e-30)) ** 0.5}


@torch.no_grad()
def snapshot_linears(model, skip=SKIP) -> dict:
    return {name: mod.weight.detach().cpu().clone() for name, mod in _targets(model, skip)}


@torch.no_grad()
def restore_linears(model, snap: dict, skip=SKIP) -> int:
    n = 0
    for name, mod in _targets(model, skip):
        if name in snap:
            mod.weight.copy_(snap[name].to(mod.weight.device, mod.weight.dtype))
            n += 1
    return n


class FakeQuant(nn.Module):
    """Parametrization putting quantize dequantize in the forward pass.

    Registered on a Linear's weight, every forward sees quantized weights while
    the stored parameter stays full precision. Gradients reach the stored
    parameter unchanged, which is the straight through estimator: the rounding
    has zero gradient almost everywhere, so pretending it is the identity on
    the backward pass is the standard and the only workable choice.
    """

    def __init__(self, bits: int = 4, group_size: int = 128, symmetric: bool = False):
        super().__init__()
        self.bits, self.group_size, self.symmetric = bits, group_size, symmetric

    def forward(self, W):
        if not torch.is_grad_enabled() or not W.requires_grad:
            return quantize_dequantize(W, self.bits, self.group_size, self.symmetric)
        return W + (quantize_dequantize(W, self.bits, self.group_size, self.symmetric) - W).detach()


def attach_fake_quant(model, bits: int = 4, group_size: int = 128,
                      symmetric: bool = False, skip=SKIP) -> int:
    """Make every decoder Linear quantization aware. Returns how many."""
    from torch.nn.utils import parametrize
    n = 0
    for _, mod in _targets(model, skip):
        parametrize.register_parametrization(
            mod, "weight", FakeQuant(bits, group_size, symmetric))
        n += 1
    return n


def detach_fake_quant(model, skip=SKIP, leave_parametrized: bool = False) -> int:
    """Remove the parametrizations.

    leave_parametrized False, the default, restores the stored full precision
    weights. That is what the merge path wants: the LoRA was trained with the
    rounding in the forward pass, but the artifact is built by merging into
    full precision and quantizing once at the end, which is how a GGUF is
    actually produced. Folding the quantized values in here instead would
    quantize twice and compound the error.

    leave_parametrized True bakes the quantized values in, for inspecting the
    exact tensors the training loop saw.
    """
    from torch.nn.utils import parametrize
    n = 0
    for _, mod in _targets(model, skip):
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(
                mod, "weight", leave_parametrized=leave_parametrized)
            n += 1
    return n


PRESETS = {
    "bf16":     {"bits": 16, "group_size": 0},
    "int8":     {"bits": 8,  "group_size": 128},   # about Q8_0
    "int4-g128": {"bits": 4, "group_size": 128},   # about Q4_K_M, AWQ, NF4
    "int4-g32": {"bits": 4,  "group_size": 32},    # about Q4_K_S with finer groups
    "int3-g128": {"bits": 3, "group_size": 128},   # about Q3_K, where things break
}


# --- does the edit survive the grid? ---------------------------------------

@torch.no_grad()
def edit_survival(W_orig: torch.Tensor, W_edit: torch.Tensor, bits: int = 4,
                  group_size: int = 128, symmetric: bool = False) -> dict:
    """How much of a weight edit is left after both models are quantized.

    The edit is delta = W_edit - W_orig. What the user actually runs is the
    difference between the two quantized models, so the question is how much of
    delta survives rounding:

        kept = Q(W_edit) - Q(W_orig)

    Two numbers. retained is the component of kept lying along delta, which is
    the fraction of the intended edit that made it through, and can exceed one
    when rounding happens to push further in the same direction. ratio is the
    raw magnitude, which also counts rounding noise pointing anywhere else.

    An edit spread thinly over many matrices is the dangerous case: if each
    weight moves less than half a quantization step, rounding erases all of it
    and the shipped model is bit identical to the unedited one.
    """
    d = (W_edit.float() - W_orig.float())
    dn = float(d.pow(2).sum())
    if dn < 1e-30:
        return {"retained": float("nan"), "ratio": float("nan"), "delta_norm": 0.0,
                "step": float("nan"), "delta_vs_step": float("nan")}
    kept = (quantize_dequantize(W_edit, bits, group_size, symmetric).float()
            - quantize_dequantize(W_orig, bits, group_size, symmetric).float())
    # typical quantization step on this tensor
    n_in = W_orig.shape[-1]
    gs = n_in if group_size in (0, -1) or group_size > n_in else group_size
    X = W_orig.float().reshape(-1, max(n_in // gs, 1), min(gs, n_in))
    step = float(((X.amax(-1) - X.amin(-1)) / (2 ** bits - 1)).mean())
    return {
        "retained": float((kept * d).sum() / dn),
        "ratio": float(kept.pow(2).sum().sqrt() / d.pow(2).sum().sqrt()),
        "delta_norm": dn ** 0.5,
        "step": step,
        "delta_vs_step": float(d.abs().mean()) / max(step, 1e-12),
    }


@torch.no_grad()
def survival_report(model, originals: dict, writers, bits: int = 4,
                    group_size: int = 128) -> dict:
    """edit_survival aggregated over every edited residual writer."""
    rows, tot_d, tot_k = [], 0.0, 0.0
    for w in writers:
        if w.name not in originals:
            continue
        W0 = originals[w.name]
        W1 = w.module.weight.detach().cpu()
        if torch.equal(W0, W1):
            continue
        s = edit_survival(W0, W1, bits, group_size)
        s["name"], s["layer"] = w.name, w.layer
        rows.append(s)
        tot_d += s["delta_norm"] ** 2
        tot_k += (s["ratio"] * s["delta_norm"]) ** 2
    if not rows:
        return {"edited_tensors": 0}
    return {
        "edited_tensors": len(rows),
        "bits": bits, "group_size": group_size,
        "retained_mean": sum(r["retained"] for r in rows) / len(rows),
        "ratio_overall": (tot_k / max(tot_d, 1e-30)) ** 0.5,
        "delta_vs_step_mean": sum(r["delta_vs_step"] for r in rows) / len(rows),
        "worst": min(rows, key=lambda r: r["retained"])["name"],
        "per_tensor": rows,
    }
