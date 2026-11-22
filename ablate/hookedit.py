"""The Stage 3 edit as forward hooks, for models too big to keep rewriting.

apply_plan() rewrites every residual writer and needs a CPU snapshot of all of
them to revert between search trials: about a quarter of the model in host RAM
plus a GPU copy per trial. On a 72B that is 35 GB of snapshot and 35 GB moved
per trial; past what fits in bf16 on the cards it is not possible at all.

HookEdit produces the same function without touching a weight. For a writer
y = W x + b and the plan's projector P = I - a Q^T Q:

  norm_restore none     W' = P W        ->  y' = P (y - b) + b
  norm_restore column   W' = (P W) D    ->  y' = P (W (D x)) + b
                        D = diag(|w_j| / |P w_j|), one scale per input column

so a pre hook scales the input by D and a forward hook projects the output.
Nothing is copied, nothing has to be restored, and it works on whatever the
module really is: a plain Linear, a Linear sharded over several GPUs, a
bitsandbytes 4 bit Linear, or one accelerate keeps on disk. The column scales
are read through the module's own forward (W = f(I) - f(0), in chunks), so they
are the norms of the weights that actually run, quantized or not.

Row restore is not available here: it is a diagonal map on the output side
that depends on every row of P W, and it is only kept in apply.py to reproduce
the reference implementation.

Mixture of experts blocks (Writer.block) are hooked on their output, which is
exact for norm_restore none. Column restore would need a different input scale
inside every expert, which no hook on the block can apply, so it is refused
there; the stream baker can still bake it, but then what is baked is not what
was scored.
"""
from __future__ import annotations

import torch

from .apply import EditPlan
from .modelio import residual_writers


def _exec_device(module) -> torch.device:
    hook = getattr(module, "_hf_hook", None)
    dev = getattr(hook, "execution_device", None)
    if dev is not None:
        return torch.device(dev)
    return module.weight.device


def _float_dtype(module) -> torch.dtype:
    dt = getattr(module, "compute_dtype", None) or module.weight.dtype
    return dt if dt.is_floating_point else torch.float32


@torch.no_grad()
def column_scales(module, Q: torch.Tensor, alpha: float, chunk: int = 4096) -> torch.Tensor:
    """D for column norm restore, [d_in] float32 on the module's device.

    |P w|^2 = |w|^2 - (2a - a^2) |Q w|^2 for orthonormal Q, so only the column
    norms and the k by d_in matrix Q W are needed, never P W itself.
    """
    dev, dt = _exec_device(module), _float_dtype(module)
    d_in = module.in_features
    Qd = Q.to(dev, torch.float32)
    zero = module(torch.zeros(1, d_in, device=dev, dtype=dt)).float()
    out = torch.empty(d_in, device=dev, dtype=torch.float32)
    c = 2.0 * alpha - alpha * alpha
    for s in range(0, d_in, chunk):
        e = min(s + chunk, d_in)
        eye = torch.zeros(e - s, d_in, device=dev, dtype=dt)
        eye[torch.arange(e - s, device=dev), torch.arange(s, e, device=dev)] = 1.0
        cols = module(eye).float() - zero            # row j is column s+j of W
        n_old = cols.norm(dim=1)
        qw = cols @ Qd.T
        n_new = (n_old.square() - c * qw.square().sum(1)).clamp(min=0).sqrt()
        out[s:e] = n_old / n_new.clamp(min=1e-12)
        del eye, cols
    return out


class HookEdit:
    """with HookEdit(model, subspace, plan): the model behaves as apply_plan would leave it."""

    def __init__(self, model, subspace, plan: EditPlan, chunk: int = 4096):
        if plan.norm_restore not in ("column", "none"):
            raise NotImplementedError("HookEdit supports norm_restore column or none")
        if plan.edit_embedding:
            raise NotImplementedError("HookEdit does not edit the embedding")
        self.model, self.subspace, self.plan, self.chunk = model, subspace, plan, chunk
        self._handles = []
        self.record: dict = {}

    def __enter__(self):
        plan, touched, skipped = self.plan, [], 0
        todo = []
        for w in residual_writers(self.model, kinds=plan.kinds):
            a = plan.alpha(w.layer)
            if a < plan.min_alpha:
                skipped += 1
                continue
            src = plan.source_layer if plan.source_layer is not None else w.layer
            src = max(0, min(int(src), self.subspace.basis.shape[0] - 1))
            Q = self.subspace.layer(src)
            if Q.shape[0] == 0:
                skipped += 1
                continue
            if w.block:
                if plan.norm_restore != "none":
                    raise NotImplementedError(
                        f"{w.name} is a mixture of experts block: HookEdit needs norm_restore none")
                D = None
            else:
                assert w.module.out_features == Q.shape[1], (
                    f"wrong orientation: {w.name} writes {w.module.out_features}, hidden {Q.shape[1]}")
                # scales are measured before any hook is on the module
                D = (column_scales(w.module, Q, a, self.chunk)
                     if plan.norm_restore == "column" else None)
            todo.append((w, a, Q, D))
            touched.append({"name": w.name, "layer": w.layer, "kind": w.kind,
                            "alpha": round(a, 4), "rank": int(Q.shape[0]), "source": src})
        for w, a, Q, D in todo:
            if D is not None:
                self._handles.append(w.module.register_forward_pre_hook(self._pre(D)))
            self._handles.append(w.module.register_forward_hook(self._post(Q, a, w.module)))
        self.record = {"edited": touched, "skipped": skipped, "n_edited": len(touched),
                       "mode": "hook"}
        return self

    @staticmethod
    def _pre(D):
        def fn(module, args):
            x = args[0]
            return (x * D.to(x.device, x.dtype),) + tuple(args[1:])
        return fn

    @staticmethod
    def _post(Q, a, module):
        def fn(mod, args, y):
            if isinstance(y, tuple):
                return (fn(mod, args, y[0]),) + tuple(y[1:])
            Qd = Q.to(y.device, y.dtype)
            b = getattr(mod, "bias", None)
            if b is not None:
                b = b.to(y.device, y.dtype)
                y0 = y - b
                return y0 - a * ((y0 @ Qd.T) @ Qd) + b
            return y - a * ((y @ Qd.T) @ Qd)
        return fn

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False
