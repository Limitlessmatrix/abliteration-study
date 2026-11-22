"""Stage 3: surgical application of a refusal subspace to the weights.

Each residual writing matrix W has shape [hidden, d_in]; its columns are the
vectors it can add to the residual stream. Projecting the columns off the
subspace is what removes the model's ability to write refusal.

Three knobs separate this from the standard recipe: a per layer strength
kernel instead of a flat edit everywhere, a restricted set of matrices, and a
column norm restore so that RMSNorm downstream sees the magnitude it was
calibrated on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch

from .modelio import residual_writers, n_layers


@dataclass
class EditPlan:
    center: float            # kernel centre, in layer index
    width: float             # gaussian sigma in layers; ignored when uniform
    peak: float              # strength at the centre, 1.0 is full projection
    kernel: str = "gaussian"  # gaussian | band | uniform | triangular
    source_layer: int | None = None   # broadcast one layer's subspace across the edit
    kinds: tuple = ("attn_out", "mlp_out")
    norm_restore: str = "column"   # column | row | none, see _project_columns
    min_alpha: float = 0.01  # below this a layer is left untouched
    edit_embedding: bool = False

    @property
    def preserve_norm(self) -> bool:
        return self.norm_restore != "none"

    def alpha(self, layer: int) -> float:
        if self.kernel == "uniform":
            a = self.peak
        elif self.kernel == "band":
            # Flat full strength inside a window, nothing outside. This is the
            # shape the published working configurations actually use: Grimjim's
            # gemma-3-12b recipe ablates layers 11 to 41 of 48 at scale 1.0.
            # A Gaussian narrow enough to look surgical turns out to be too weak
            # to move refusal at all.
            a = self.peak if abs(layer - self.center) <= self.width else 0.0
        elif self.kernel == "triangular":
            a = self.peak * max(0.0, 1.0 - abs(layer - self.center) / max(self.width, 1e-6))
        else:
            s = max(self.width, 1e-6)
            a = self.peak * math.exp(-((layer - self.center) ** 2) / (2 * s * s))
        return float(max(0.0, min(a, 1.0)))


def _project_columns(W: torch.Tensor, Q: torch.Tensor, alpha: float,
                     norm_restore: str = "column") -> torch.Tensor:
    """W [d_out=hidden, d_in], Q [k, hidden] orthonormal rows.

    The projector left multiplies: W' = W - alpha * Q^T Q W. That acts on the
    first index, so it is the COLUMNS of W that are vectors in residual space,
    and each one is what gets projected.

    norm_restore decides what is scaled back afterwards so that RMSNorm sees
    the magnitude it was calibrated on:

      column  rescale each column to its original norm. Every column stays
              orthogonal to the subspace, so the ablation survives exactly and
              the Frobenius norm is restored.
      row     rescale each row, which is what math.md section 6 and the skill's
              reference apply_edit.py do. A per output coordinate rescale is a
              diagonal map, and a diagonal map does not commute with the
              projector, so this puts a measurable part of the ablated
              component back. Measured on a random [512, 1024] matrix with
              k = 2: output magnitude along the subspace is exactly 0 for
              column restore and 0.0077 for row restore against 1.8184
              unablated, with both restoring the Frobenius norm to 1.0000.
              Kept only for reproducing the reference implementation.
      none    no restore.
    """
    if alpha <= 0 or Q.shape[0] == 0:
        return W
    assert W.shape[0] == Q.shape[1], (
        f"wrong orientation: W rows {W.shape[0]} != hidden {Q.shape[1]}")
    Qd = Q.to(device=W.device, dtype=torch.float32)
    Wf = W.float()
    out = Wf - alpha * (Qd.T @ (Qd @ Wf))
    if norm_restore == "column":
        n_old = Wf.norm(dim=0, keepdim=True)
        out = out * (n_old / out.norm(dim=0, keepdim=True).clamp(min=1e-12))
    elif norm_restore == "row":
        n_old = Wf.norm(dim=1, keepdim=True)
        out = out * (n_old / out.norm(dim=1, keepdim=True).clamp(min=1e-12))
    elif norm_restore != "none":
        raise ValueError(norm_restore)
    return out.to(W.dtype)


@torch.no_grad()
def apply_plan(model, subspace, plan: EditPlan,
               originals: dict[str, torch.Tensor] | None = None) -> dict:
    """Edit the model in place. Returns a record of what was touched.

    originals maps writer name -> a CPU float32 copy of the untouched weight.
    Pass the dict returned by snapshot() to make repeated applications during a
    search start from the original weights every time.
    """
    touched, skipped = [], 0
    writers = residual_writers(model, kinds=plan.kinds)
    _no_blocks(writers)
    for w in writers:
        a = plan.alpha(w.layer)
        base = None
        if originals is not None and w.name in originals:
            base = originals[w.name].to(w.module.weight.device, w.module.weight.dtype)
        if a < plan.min_alpha:
            if base is not None:
                w.module.weight.copy_(base)
            skipped += 1
            continue
        # A single mid-network direction broadcast across the band is what the
        # reference implementations do, and it beats per-layer directions in
        # practice even though per-layer is the more principled story.
        src_layer = plan.source_layer if plan.source_layer is not None else w.layer
        src_layer = max(0, min(int(src_layer), subspace.basis.shape[0] - 1))
        Q = subspace.layer(src_layer)
        if Q.shape[0] == 0:
            skipped += 1
            continue
        W_src = base if base is not None else w.module.weight.data
        w.module.weight.copy_(_project_columns(W_src, Q, a, plan.norm_restore))
        touched.append({"name": w.name, "layer": w.layer, "kind": w.kind,
                        "alpha": round(a, 4), "rank": int(Q.shape[0]),
                        "source": src_layer})

    if plan.edit_embedding:
        emb = model.get_input_embeddings().weight
        l = int(round(plan.center))
        Q = subspace.layer(max(0, min(l, subspace.basis.shape[0] - 1)))
        if Q.shape[0]:
            # embedding rows are residual vectors, so project rows not columns
            e = _project_columns(emb.data.T, Q, plan.peak, plan.norm_restore).T
            emb.copy_(e)
            touched.append({"name": "embed_tokens", "layer": -1, "kind": "embed",
                            "alpha": plan.peak, "rank": int(Q.shape[0])})

    return {"plan": asdict(plan), "edited": touched, "skipped": skipped,
            "n_edited": len(touched)}


def _no_blocks(writers):
    if any(w.block for w in writers):
        raise NotImplementedError(
            "this model writes its MLP output through mixture of experts blocks; edit it with "
            "--edit-mode hook (norm_restore none) and bake with scripts/bake_stream.py")


@torch.no_grad()
def snapshot(model, kinds=("attn_out", "mlp_out")) -> dict[str, torch.Tensor]:
    """CPU copies of every writer, in the model dtype, so a search can revert
    cheaply. An 8B pair of o_proj plus down_proj is about 5 GB in bfloat16."""
    writers = residual_writers(model, kinds=kinds)
    _no_blocks(writers)
    return {w.name: w.module.weight.detach().cpu().clone() for w in writers}


@torch.no_grad()
def restore(model, originals: dict[str, torch.Tensor],
            kinds=("attn_out", "mlp_out")) -> int:
    n = 0
    for w in residual_writers(model, kinds=kinds):
        if w.name in originals:
            w.module.weight.copy_(originals[w.name].to(w.module.weight.device,
                                                       w.module.weight.dtype))
            n += 1
    return n


def naive_plan(model, subspace) -> EditPlan:
    """The standard recipe, for the side by side comparison.

    Flat full strength projection on every layer, every residual writer and the
    embedding matrix, no norm restore. This is what FailSpy's abliterator and
    most published abliterated checkpoints do.
    """
    L = n_layers(model)
    return EditPlan(center=(L - 1) / 2, width=float(L), peak=1.0, kernel="uniform",
                    kinds=("attn_out", "mlp_out"), norm_restore="none",
                    min_alpha=0.0, edit_embedding=True)


# --- conditional ablation at inference time -------------------------------

class ConditionalAblation:
    """Stage 3 alternative: ablate only on sequences the gate says are harmful.

    The gate is per sequence, not per token, and it is driven by a separate
    harmfulness recognition direction rather than by the refusal subspace being
    ablated. Two reasons. Whether a request is harmful is a property of the
    whole instruction, not of individual positions, so a per token gate flickers
    mid sentence and produces exactly the incoherence the coherence judge is
    there to catch. And gating on the thing you are removing is circular: once
    ablation starts suppressing the refusal projection the gate stops firing.

    Score is the mean activation over the instruction tokens at the condition
    layer, projected onto the condition vector. The gate latches at prefill and
    holds for the whole generation.

    tau is calibrated to a target false trigger rate on harmless sequences, so
    KL on anything that does not trip the gate is exactly zero by construction.
    """

    def __init__(self, model, subspace, plan: EditPlan, weight: torch.Tensor,
                 bias: float, tau: float, condition_layer: int, n_suffix: int = 1):
        self.model = model
        self.subspace = subspace
        self.plan = plan
        # The gate is an affine probe, score = pooled @ w + b, not a dot product
        # with a difference of means. Gemma style architectures carry a handful
        # of outlier dimensions two orders of magnitude larger than the rest,
        # and a raw projection is decided entirely by those, which is how a
        # harmful mean of -136 ends up sitting below a harmless threshold of
        # +827. A fitted probe absorbs the per dimension scale.
        self.w = weight.float()
        self.bias = float(bias)
        self.tau = float(tau)
        self.condition_layer = int(condition_layer)
        self.n_suffix = max(int(n_suffix), 1)
        self._handles = []
        self._mask = None
        self._gate = None
        self.n_seq = 0
        self.n_fired = 0
        self.scores: list[float] = []

    # the decoder layers never see the attention mask, so grab it on the way in
    def _pre(self, module, args, kwargs):
        am = kwargs.get("attention_mask")
        if am is None and len(args) > 1:
            am = args[1]
        self._mask = am if isinstance(am, torch.Tensor) and am.dim() == 2 else None
        return None

    def _cond_hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] <= 1:
            return output                      # decode step, gate already latched
        v = self.w.to(h.device, h.dtype)
        m = self._mask
        if m is None or m.shape[-1] != h.shape[1]:
            m = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        m = m.to(h.device, h.dtype).clone()
        m[:, -self.n_suffix:] = 0              # instruction tokens only
        denom = m.sum(1, keepdim=True).clamp(min=1.0)
        pooled = (h * m.unsqueeze(-1)).sum(1) / denom
        score = pooled @ v + self.bias
        self._gate = score > self.tau
        self.n_seq += int(score.numel())
        self.n_fired += int(self._gate.sum())
        self.scores.extend(score.float().cpu().tolist())
        return output

    def _ablate_hook(self, layer_idx):
        def fn(module, args, output):
            if self._gate is None or not bool(self._gate.any()):
                return output
            a = self.plan.alpha(layer_idx)
            if a < self.plan.min_alpha:
                return output
            Q = self.subspace.layer(layer_idx)
            if Q.shape[0] == 0:
                return output
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            Qd = Q.to(h.device, h.dtype)
            g = self._gate.to(h.device, h.dtype).view(-1, 1, 1)
            h = h - a * g * ((h @ Qd.T) @ Qd)
            return ((h,) + tuple(output[1:])) if is_tuple else h
        return fn

    def reset(self):
        self._gate = None
        self.n_seq = self.n_fired = 0
        self.scores = []

    def __enter__(self):
        from .modelio import decoder_layers
        self.reset()
        self._handles.append(
            self.model.register_forward_pre_hook(self._pre, with_kwargs=True))
        layers = decoder_layers(self.model)
        self._handles.append(
            layers[self.condition_layer].register_forward_hook(self._cond_hook))
        for i, layer in enumerate(layers):
            if i > self.condition_layer and self.plan.alpha(i) >= self.plan.min_alpha:
                self._handles.append(layer.register_forward_hook(self._ablate_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._gate = self._mask = None
        return False

    @property
    def trigger_rate(self) -> float:
        return self.n_fired / max(self.n_seq, 1)


@torch.no_grad()
def condition_pooled(model, tok, prompts: list[str], condition_layer: int,
                     n_suffix: int = 1, batch_size: int = 8,
                     device: str = "cuda") -> torch.Tensor:
    """Mean activation over instruction tokens at the condition layer, [n, H]."""
    from .modelio import batched, format_chat, input_device

    out = []
    for group in batched(prompts, batch_size):
        texts = [format_chat(tok, p) for p in group]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(input_device(model))
        res = model(**enc, output_hidden_states=True, use_cache=False)
        h = res.hidden_states[condition_layer + 1]
        m = enc["attention_mask"].to(h.dtype).clone()
        m[:, -max(n_suffix, 1):] = 0
        pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        out.append(pooled.float().cpu())
        del res
    return torch.cat(out)


def condition_scores(model, tok, prompts: list[str], condition: torch.Tensor,
                     condition_layer: int, n_suffix: int = 1,
                     batch_size: int = 8, device: str = "cuda",
                     center: torch.Tensor | None = None) -> torch.Tensor:
    """Per sequence gate score, the same quantity the hook computes."""
    v = condition / condition.norm().clamp(min=1e-12)
    pooled = condition_pooled(model, tok, prompts, condition_layer,
                              n_suffix=n_suffix, batch_size=batch_size, device=device)
    if center is not None:
        pooled = pooled - center.float()
    return pooled @ v.float()


def tau_for_fpr(harmless_scores: torch.Tensor, target_fpr: float = 0.01) -> float:
    """Threshold admitting at most target_fpr of harmless sequences."""
    q = 1.0 - float(target_fpr)
    return float(harmless_scores.float().quantile(min(max(q, 0.0), 1.0)))


def fit_condition_probe(pooled_harmful: torch.Tensor, pooled_harmless: torch.Tensor,
                        seed: int = 0, C: float = 1.0) -> dict:
    """Affine harmfulness recognition probe on pooled instruction activations.

    Standardisation is folded back into the returned weight and bias, so the
    hook applies one dot product and an add with no further bookkeeping:

        ((x - mu) / sd) @ w + b  ==  x @ (w / sd) + (b - (mu / sd) @ w)

    Returns weight, bias and the cross validated AUC, which is the number that
    says whether the gate is worth shipping at all.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = torch.cat([pooled_harmful, pooled_harmless]).float()
    y = np.array([1] * pooled_harmful.shape[0] + [0] * pooled_harmless.shape[0])
    mu = X.mean(0)
    sd = X.std(0).clamp(min=1e-6)
    Xs = ((X - mu) / sd).numpy().astype("float64")

    n_min = int(min((y == 0).sum(), (y == 1).sum()))
    auc = float("nan")
    if n_min >= 2:
        cv = StratifiedKFold(n_splits=max(2, min(5, n_min)), shuffle=True, random_state=seed)
        auc = float(cross_val_score(
            LogisticRegression(max_iter=2000, C=C, class_weight="balanced"),
            Xs, y, cv=cv, scoring="roc_auc").mean())

    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    clf.fit(Xs, y)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    b = float(clf.intercept_[0])
    w_eff = w / sd
    b_eff = b - float((mu / sd) @ w)
    return {"weight": w_eff, "bias": b_eff, "auc": auc,
            "n_harmful": int(pooled_harmful.shape[0]),
            "n_harmless": int(pooled_harmless.shape[0])}


def probe_scores(pooled: torch.Tensor, probe: dict) -> torch.Tensor:
    return pooled.float() @ probe["weight"] + probe["bias"]
