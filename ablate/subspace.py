"""Stage 2: a clean, protected refusal subspace.

The pipeline, per layer:
  1. SVD over per pair activation differences gives a rank k refusal subspace
     rather than a single difference of means direction. k > 1 is the concept
     cone of Wollschlaeger et al. 2025.
  2. The subspace is projected off the harmless mean activation, so ablation
     does not shift the representation of inputs that were never going to be
     refused.
  3. A direction per calibration behaviour is extracted by contrasting probe
     against matched control, and the refusal subspace is projected off their
     span. This is the protection step.
  4. Linear probes separate the refusal decision from harmfulness recognition,
     following Zhao et al. 2025, so the subspace can be built from refuse
     against comply rather than harmful against harmless.

All functions take and return float32 CPU tensors shaped [layers, k, hidden].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _orthonormalize(B: torch.Tensor, tol: float = 1e-6,
                    min_row: float = 1e-3) -> torch.Tensor:
    """Row-wise orthonormal basis of the row space of B [k, H].

    Rows are assumed to have started at unit norm, so min_row is absolute. Rows
    whose length a projection has already consumed are dropped before the QR
    rather than after it. That ordering is load bearing: QR renormalises a
    near zero row back to unit length, and what it renormalises is the
    numerical noise left behind by the projection, which puts the component
    that was just removed straight back into the basis.
    """
    if B.shape[0] == 0:
        return B
    B = B[B.norm(dim=1) > min_row]
    if B.shape[0] == 0:
        return B
    Q, R = torch.linalg.qr(B.T)                     # [H, k], [k, k]
    keep = R.diagonal().abs() > tol * R.diagonal().abs().max().clamp(min=1e-12)
    return Q.T[keep].contiguous()


def _project_off(B: torch.Tensor, S: torch.Tensor, passes: int = 2) -> torch.Tensor:
    """Project the rows of B onto the orthogonal complement of span(S), then
    re-orthonormalise, as math.md sections 3 and 4 require after every
    projection.

    Two passes. Re-orthonormalising after the first projection mixes the rows
    against each other, and that mixing can reintroduce a little of what was
    removed, so the strip is repeated and the result settles at machine
    precision.
    """
    if S is None or S.numel() == 0 or S.shape[0] == 0:
        return _orthonormalize(B)
    Qs = _orthonormalize(S)
    if Qs.shape[0] == 0:
        return _orthonormalize(B)
    for _ in range(passes):
        B = B - (B @ Qs.T) @ Qs
        B = _orthonormalize(B)
        if B.shape[0] == 0:
            break
    return B


@dataclass
class Subspace:
    """Per layer orthonormal refusal basis and the diagnostics behind it."""
    basis: torch.Tensor                      # [L, k, H]
    variance_ratio: torch.Tensor             # [L, k] share of difference energy
    separation: torch.Tensor                 # [L] cohen d of the top direction
    rank: torch.Tensor                       # [L] rows surviving protection
    protected_overlap: dict = field(default_factory=dict)   # name -> [L, k] cosine
    meta: dict = field(default_factory=dict)

    max_rank: int | None = None

    def layer(self, l: int) -> torch.Tensor:
        """Live rows for layer l, [k_l, H], clamped by max_rank when set."""
        r = int(self.rank[l])
        if self.max_rank is not None:
            r = min(r, int(self.max_rank))
        return self.basis[l][:r]

    def with_rank(self, k: int) -> "Subspace":
        """A view using only the top k directions. Lets one extraction serve a
        whole Optuna sweep over subspace rank without recomputing the SVD."""
        import copy
        o = copy.copy(self)
        o.max_rank = k
        return o

    def best_layer(self, lo: float = 0.25, hi: float = 0.85) -> int:
        """Layer with the strongest separation inside the usual depth window.

        Refusal is decided in the middle of the stack. Early layers carry token
        identity and the last couple carry output formatting; editing either
        does damage out of proportion to the refusal it removes.
        """
        L = self.separation.shape[0]
        a, b = int(lo * L), max(int(hi * L), int(lo * L) + 1)
        window = self.separation.clone()
        window[:a] = -torch.inf
        window[b:] = -torch.inf
        return int(window.argmax())


def difference_subspace(pos: torch.Tensor, neg: torch.Tensor, k: int = 2,
                        paired: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD of activation differences at every layer.

    pos, neg: [n, L, H]. Returns (basis [L,k,H], variance_ratio [L,k],
    separation [L]). Row signs are fixed so each direction points from neg
    towards pos.
    """
    assert pos.shape[1:] == neg.shape[1:], "layer/hidden mismatch"
    L, H = pos.shape[1], pos.shape[2]
    k = min(k, H)
    basis = torch.zeros(L, k, H)
    var = torch.zeros(L, k)
    sep = torch.zeros(L)
    for l in range(L):
        P, Ng = pos[:, l, :].double(), neg[:, l, :].double()
        mu = P.mean(0) - Ng.mean(0)
        if paired:
            n = min(P.shape[0], Ng.shape[0])
            D = P[:n] - Ng[:n]
        else:
            D = P - Ng.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        kk = min(k, Vh.shape[0])
        V = Vh[:kk]
        flip = torch.sign(V @ mu).unsqueeze(1)
        flip[flip == 0] = 1.0
        V = V * flip
        basis[l, :kk] = V.float()
        tot = (S ** 2).sum().clamp(min=1e-30)
        var[l, :kk] = ((S[:kk] ** 2) / tot).float()
        # cohen d of the projection onto the leading direction
        v = V[0]
        a, b = P @ v, Ng @ v
        pooled = torch.sqrt((a.var(unbiased=True) + b.var(unbiased=True)) / 2).clamp(min=1e-12)
        sep[l] = float(((a.mean() - b.mean()) / pooled).abs())
    return basis, var, sep


def behaviour_directions(probe: torch.Tensor, control: torch.Tensor,
                         paired: bool = True) -> torch.Tensor:
    """One normalised direction per layer from a matched probe/control set.

    The top right singular vector of the per pair difference, not the
    difference of means, for the same reason the refusal subspace uses SVD:
    the mean is only the leading component when every pair agrees, and these
    pairs do not all agree.

    probe, control: [n, L, H] -> [L, H].
    """
    L, H = probe.shape[1], probe.shape[2]
    out = torch.zeros(L, H)
    for l in range(L):
        P, C = probe[:, l, :].double(), control[:, l, :].double()
        mu = P.mean(0) - C.mean(0)
        if paired:
            n = min(P.shape[0], C.shape[0])
            D = P[:n] - C[:n]
        else:
            D = P - C.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(D, full_matrices=False)
        v = Vh[0]
        if float(v @ mu) < 0:
            v = -v
        out[l] = (v / v.norm().clamp(min=1e-12)).float()
    return out


def build(pos: torch.Tensor, neg: torch.Tensor, k: int = 2, paired: bool = True,
          harmless_mean: torch.Tensor | None = None,
          protected: dict[str, torch.Tensor] | None = None,
          protect: bool = True) -> Subspace:
    """Stages 2.1 through 2.3 in one call.

    harmless_mean: [L, H] mean activation of the harmless set.
    protected:     name -> [L, H] behaviour direction to preserve.
    """
    basis, var, sep = difference_subspace(pos, neg, k=k, paired=paired)
    L, kk, H = basis.shape
    protected = protected or {}

    overlap = {name: torch.zeros(L, kk) for name in protected}
    for name, d in protected.items():
        dn = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        overlap[name] = torch.einsum("lkh,lh->lk", basis, dn)

    # math.md sections 3 and 4 run these as two separate projections, each
    # followed by its own re-orthonormalisation, not as one combined strip.
    names = sorted(protected)
    rank = torch.full((L,), kk, dtype=torch.int32)
    out = basis.clone()
    for l in range(L):
        B = basis[l]
        if harmless_mean is not None:
            m = harmless_mean[l]
            B = _project_off(B, (m / m.norm().clamp(min=1e-12)).unsqueeze(0))
        if protect and names:
            P = torch.stack([protected[n][l] for n in names])
            B = _project_off(B, P)
        if harmless_mean is None and not (protect and names):
            B = _orthonormalize(B)
        rank[l] = B.shape[0]
        out[l] = 0.0
        if B.shape[0]:
            out[l, : B.shape[0]] = B

    return Subspace(
        basis=out, variance_ratio=var, separation=sep, rank=rank,
        protected_overlap=overlap,
        meta={"k": kk, "paired": paired, "protect": protect,
              "orthogonalised_against_harmless_mean": harmless_mean is not None,
              "protected_names": sorted(protected)},
    )


def probe_auc(acts: torch.Tensor, labels: torch.Tensor, layer: int,
              folds: int = 5, seed: int = 0) -> float:
    """Cross validated AUC of a linear probe on layer activations.

    Used to confirm that a subspace carries the refusal decision and not just
    harmfulness recognition: fit on refuse against comply and on harmful
    against harmless separately and compare.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import numpy as np

    X = acts[:, layer, :].numpy().astype("float64")
    y = labels.numpy().astype(int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    n_min = min((y == 0).sum(), (y == 1).sum())
    folds = max(2, min(folds, int(n_min)))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(scores))


def best_layer_by_probe(acts: torch.Tensor, refused: torch.Tensor,
                        lo: float = 0.25, hi: float = 0.85,
                        stride: int = 1, seed: int = 0) -> dict:
    """Pick the application layer by refusal probe accuracy, per math.md 5.

    The layer with the largest singular value is not usually the layer where
    the refusal decision is most linearly available. Returns the best layer
    and the per layer AUC so the choice is inspectable.
    """
    L = acts.shape[1]
    a, b = int(lo * L), max(int(hi * L), int(lo * L) + 1)
    aucs = {}
    for l in range(a, b, stride):
        try:
            aucs[l] = probe_auc(acts, refused, l, seed=seed)
        except ImportError:
            return {"error": "scikit-learn not installed"}
    live = {k: v for k, v in aucs.items() if v == v}
    if not live:
        return {"error": "probe failed at every layer", "auc": aucs}
    best = max(live, key=live.get)
    return {"best_layer": best, "auc": live[best],
            "per_layer": {int(k): round(float(v), 4) for k, v in sorted(aucs.items())}}


def report(ss: Subspace, top: int = 8) -> str:
    L = ss.separation.shape[0]
    bl = ss.best_layer()
    lines = [
        f"layers {L}  k {ss.meta.get('k')}  protect {ss.meta.get('protect')}",
        f"best layer {bl}  separation {ss.separation[bl]:.2f}  surviving rank {int(ss.rank[bl])}",
        f"variance explained at best layer {[round(float(x),3) for x in ss.variance_ratio[bl]]}",
    ]
    if ss.protected_overlap:
        lines.append("cosine of top refusal direction with each protected behaviour, at best layer:")
        for name, ov in sorted(ss.protected_overlap.items()):
            lines.append(f"  {name:22s} {float(ov[bl, 0]):+.3f}")
    order = ss.separation.argsort(descending=True)[:top]
    lines.append("strongest layers: " + ", ".join(
        f"{int(l)}({float(ss.separation[l]):.2f})" for l in order))
    return "\n".join(lines)
