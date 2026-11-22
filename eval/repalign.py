"""Representation alignment between a base model and an edited one: a second damage measure beside KL.

Why (added 2026-11-16): KL compares output distributions. Two models can agree on next-token probabilities for
the reference answers while their internal representations have moved, and the reverse. The measures here are the
ones used to compare representations across models in the Platonic Representation Hypothesis work (mutual k-NN:
Huh et al. 2024, arXiv 2405.07987; reused in Balestriero and LeCun 2026, arXiv 2610.39302) plus linear CKA.

Input: for the SAME prompts, one vector per prompt from each model (n x d matrices, rows aligned). In the bake-off
these are hidden states of the 300 held-out harmless prompts, teacher-forced on the base model's own answer:
  "last" = the state at the last prompt token (where generation starts)
  "mean" = the mean state over the answer tokens
taken at four depths (25, 50, 75, 100 percent of the layers).

  lin_cka(X, Y)        1.0 = same geometry up to rotation and scale; near 0 = unrelated
  mutual_knn(X, Y, k)  mean share of each prompt's k nearest neighbours (cosine) that the two models agree on
  paired_cos(X, Y)     per-prompt cosine between the two models' vectors (only meaningful for the same network
                       before and after an edit: the coordinates are shared). Per-prompt, so it supports paired tests.
All three are 1.0 for identical inputs. CKA and mutual k-NN ignore rotations; paired_cos does not.

  python repalign.py --selftest
"""
import sys

import torch


def _center(X):
    X = X.double()
    return X - X.mean(0, keepdim=True)


def lin_cka(X, Y) -> float:
    X, Y = _center(X), _center(Y)
    num = (X.T @ Y).pow(2).sum()
    den = (X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()
    return float(num / den) if float(den) > 0 else 0.0


def _knn(X, k):
    X = torch.nn.functional.normalize(_center(X), dim=1)
    S = X @ X.T
    S.fill_diagonal_(-2.0)
    return S.topk(k, dim=1).indices


def mutual_knn(X, Y, k: int = 10) -> float:
    k = min(k, X.shape[0] - 1)
    a, b = _knn(X, k), _knn(Y, k)
    hits = [len(set(a[i].tolist()) & set(b[i].tolist())) / k for i in range(X.shape[0])]
    return sum(hits) / len(hits)


def paired_cos(X, Y):
    return torch.nn.functional.cosine_similarity(X.double(), Y.double(), dim=1).tolist()


def layer_picks(n_layers: int):
    """Indices into hidden_states (0 = embeddings, n_layers = last block) at 25 / 50 / 75 / 100 percent depth."""
    return sorted({max(1, round(n_layers * f)) for f in (0.25, 0.5, 0.75, 1.0)})


def summarise(base: dict, cont: dict, k: int = 10) -> dict:
    """base / cont: {"layer:pool": n x d tensor}. Returns {"layer:pool": {"cka", "mknn", "cos_mean", "cos_min"}}."""
    out = {}
    for key in sorted(base, key=lambda s: (int(s.split(":")[0]), s)):
        if key not in cont or base[key].shape != cont[key].shape:
            continue
        cs = paired_cos(base[key], cont[key])
        out[key] = {"cka": round(lin_cka(base[key], cont[key]), 6), "mknn": round(mutual_knn(base[key], cont[key], k), 6),
                    "cos_mean": round(sum(cs) / len(cs), 6), "cos_min": round(min(cs), 6)}
    return out


def selftest() -> int:
    g = torch.Generator().manual_seed(0)
    n, d = 120, 64
    # clustered data, so neighbourhoods are meaningful
    centres = torch.randn(8, d, generator=g) * 4
    X = centres[torch.arange(n) % 8] + torch.randn(n, d, generator=g)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    R = torch.randn(n, d, generator=g)
    small = X + 0.05 * torch.randn(n, d, generator=g)
    big = X + 3.0 * torch.randn(n, d, generator=g)
    checks = [
        ("identical: cka 1", abs(lin_cka(X, X) - 1) < 1e-9),
        ("identical: mknn 1", mutual_knn(X, X) == 1.0),
        ("identical: cos 1", min(paired_cos(X, X)) > 1 - 1e-9),
        ("rotation: cka 1", abs(lin_cka(X, X @ Q) - 1) < 1e-6),
        ("rotation: mknn 1", mutual_knn(X, X @ Q) > 0.999),
        ("rotation: cos is not 1", sum(paired_cos(X, X @ Q)) / n < 0.5),
        ("scale and shift: cka 1", abs(lin_cka(X, 3 * X + 7) - 1) < 1e-6),
        ("unrelated: cka low", lin_cka(X, R) < 0.6),
        ("unrelated: mknn low", mutual_knn(X, R) < 0.3),
        ("small noise beats big noise (cka)", lin_cka(X, small) > lin_cka(X, big)),
        ("small noise beats big noise (mknn)", mutual_knn(X, small) > mutual_knn(X, big)),
        ("small noise beats big noise (cos)", sum(paired_cos(X, small)) > sum(paired_cos(X, big))),
        ("layer picks 64", layer_picks(64) == [16, 32, 48, 64]),
        ("layer picks 3", layer_picks(3) == [1, 2, 3]),
        ("summarise keys", set(summarise({"16:last": X, "16:mean": X}, {"16:last": small, "16:mean": big})) == {"16:last", "16:mean"}),
        ("float16 input", abs(lin_cka(X.half(), X.half()) - 1) < 1e-6),
    ]
    bad = [name for name, ok in checks if not ok]
    for name in bad:
        print("SELFTEST FAIL:", name)
    print(f"repalign selftest: {len(checks) - len(bad)}/{len(checks)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
