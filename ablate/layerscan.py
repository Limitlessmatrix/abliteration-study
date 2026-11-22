"""Interventional layer selection.

A probe says where refusal is linearly readable. That is not the same question
as where ablating it actually works, and the two answers differ because
readability and causal load are different properties of a representation. So
pick the layer by intervening: ablate the candidate subspace at one layer at a
time and measure what it does to the model.

Two scores per layer, both on the first generated token under the chat
template, which costs one forward pass each and no generation:

  refusal_logprob  log mass on refusal openers over validation harmful prompts.
                   Lower is better, this is what the edit is for.
  first_token_kl   KL from the untouched model on harmless prompts. Higher is
                   damage.

The chosen layer minimises refusal_logprob + lambda * first_token_kl.
"""
from __future__ import annotations

import torch

from .capture import first_token_stats, refusal_token_ids
from .modelio import decoder_layers
from .quant import PRESETS, quantize_model_, restore_linears, snapshot_linears


class _AblateAt:
    """Ablate one layer's output, for the duration of a with block."""

    def __init__(self, model, subspace, layer: int, alpha: float = 1.0):
        self.model, self.subspace, self.layer, self.alpha = model, subspace, layer, alpha
        self._h = None

    def __enter__(self):
        Q = self.subspace.layer(self.layer)

        def fn(module, args, output):
            if Q.shape[0] == 0:
                return output
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            Qd = Q.to(h.device, h.dtype)
            h = h - self.alpha * ((h @ Qd.T) @ Qd)
            return ((h,) + tuple(output[1:])) if is_tuple else h

        self._h = decoder_layers(self.model)[self.layer].register_forward_hook(fn)
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()
        return False


@torch.no_grad()
def scan(model, tok, subspace, harmful: list[str], harmless: list[str],
         lo: float = 0.15, hi: float = 0.80, stride: int = 1,
         alpha: float = 1.0, kl_budget: float = 0.25, batch_size: int = 8,
         device: str = "cuda", verbose: bool = True,
         quant: str | None = None) -> dict:
    """-> {'best_layer', 'kl_budget', 'per_layer': {l: {...}}}

    Selection is constrained, not scalarised. Refusal log probability moves over
    about twelve nats while first token KL moves over four orders of magnitude,
    so any fixed weighting between them is really just a choice of which axis to
    ignore, and summing them picks whichever layer happens to break the model
    hardest. Instead: take the largest refusal drop among layers whose damage
    stays inside the budget, and fall back to the least damaging layer if none
    qualify.
    """
    L = len(decoder_layers(model))
    rid = refusal_token_ids(tok)

    lin = None
    if quant:
        if quant not in PRESETS:
            raise ValueError(f"unknown quant preset {quant}")
        lin = snapshot_linears(model)
        st = quantize_model_(model, **PRESETS[quant])
        if verbose:
            print(f"  scanning at {quant}, rel weight rmse {st['rel_weight_rmse']:.5f}",
                  flush=True)

    base_h = first_token_stats(model, tok, harmful, rid, batch_size=batch_size, device=device)
    base_c = first_token_stats(model, tok, harmless, rid, batch_size=batch_size, device=device)
    ref_lp = base_c.pop("_logprobs")
    base_h.pop("_logprobs", None)

    a, b = int(lo * L), max(int(hi * L), int(lo * L) + 1)
    per = {}
    for l in range(a, b, stride):
        if subspace.layer(l).shape[0] == 0:
            continue
        with _AblateAt(model, subspace, l, alpha):
            h = first_token_stats(model, tok, harmful, rid, batch_size=batch_size, device=device)
            c = first_token_stats(model, tok, harmless, rid, batch_size=batch_size,
                                  device=device, ref_logprobs=ref_lp)
        h.pop("_logprobs", None); c.pop("_logprobs", None)
        rec = {
            "refusal_logprob": h["refusal_logprob_mean"],
            "refusal_drop": base_h["refusal_logprob_mean"] - h["refusal_logprob_mean"],
            "first_token_kl": c.get("first_token_kl", 0.0),
        }
        rec["within_budget"] = rec["first_token_kl"] <= kl_budget
        per[l] = rec
        if verbose:
            print(f"  layer {l:3d}  refusal_lp {rec['refusal_logprob']:+.3f} "
                  f"(drop {rec['refusal_drop']:+.3f})  ft_kl {rec['first_token_kl']:.4f}"
                  f"{'' if rec['within_budget'] else '   over budget'}", flush=True)

    if not per:
        if lin is not None:
            restore_linears(model, lin)
        return {"error": "every candidate layer had an empty subspace"}
    if lin is not None:
        restore_linears(model, lin)

    ok = [l for l in per if per[l]["within_budget"]]
    if ok:
        best = max(ok, key=lambda l: per[l]["refusal_drop"])
    else:
        best = min(per, key=lambda l: per[l]["first_token_kl"])
    return {"best_layer": best, "kl_budget": kl_budget, "alpha": alpha,
            "scanned_at": quant or "bf16", "n_within_budget": len(ok),
            "baseline_refusal_logprob": base_h["refusal_logprob_mean"],
            "per_layer": per, "n_harmful": len(harmful), "n_harmless": len(harmless)}
