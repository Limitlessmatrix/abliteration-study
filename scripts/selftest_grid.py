"""CPU self-test of the grid-started search (ablate.search.rank1_grid / frontier). Synthetic evaluate(), no model.

Checks: the first 24 trials are exactly the grid cells in order, the sampler takes over after them, the grid
frontier is recorded, a resumed journal study neither repeats nor loses grid cells, a study with only one subspace
drops the cells of the other, runs without --grid are unchanged, and the grid start makes three seeds agree on a
problem whose optimum is a narrow rank 1 band (the failure seen on Qwen3-8B).
"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ablate.search as S


class FakeSS:
    def with_rank(self, k): self.k = k; return self

class FakeHook:
    cur = None
    def __init__(self, model, ss, plan): FakeHook.cur = (ss.k, plan); self.record = {"n_edited": 10}
    def __enter__(self): return self
    def __exit__(self, *a): return False

L = 36
def fake_eval(model, tok, sets, **kw):
    """Refusal goes away only for a rank 1 band near 0.6 depth; damage grows with width, rank and peak.
    A narrow optimum that random search finds late, like the Qwen3-8B frontier."""
    k, plan = FakeHook.cur
    c, w = plan.center / (L - 1), plan.width / (L - 1)
    hit = max(0.0, 1.0 - abs(c - 0.60) / 0.12) * (1.0 if plan.kernel == "band" else 0.5)
    strength = plan.peak * hit * min(1.0, w / 0.15) * (1.0 if plan.kinds == ("attn_out", "mlp_out") else 0.5)
    refusal = max(0.0, 0.95 - 1.05 * strength)
    kl = 0.004 + 0.05 * w * plan.peak ** 2 * (1 + 2.0 * (k - 1))
    return {"refusal_rate": refusal, "kl_mean": kl, "coherence_rate": 1.0, "over_refusal_rate": 0.0, "calibration": {}}

S.HookEdit, S.evaluate = FakeHook, fake_eval
S.objective = lambda b, r, lambda_kl=1.0, lambda_cal=2.0: {"value": r["refusal_rate"] + r["kl_mean"], "terms": {
    "refusal_rate": r["refusal_rate"], "kl": r["kl_mean"], "calibration_drop": 0.0, "over_refusal_delta": 0.0, "coherence_drop": 0.0}}
import ablate.modelio as M
M.n_layers = lambda m: L; M.has_block_writers = lambda m: False
base = {"refusal_rate": 0.95}
both = lambda: {"protected": FakeSS(), "plain": FakeSS()}
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

G = S.rank1_grid()
check("grid has 24 distinct cells inside the search space", len(G) == 24 and len({tuple(sorted(g.items())) for g in G}) == 24
      and all(S.SPACE["center_frac"][0] <= g["center_frac"] <= S.SPACE["center_frac"][1]
              and S.SPACE["width_frac"][0] <= g["width_frac"] <= S.SPACE["width_frac"][1]
              and S.SPACE["peak"][0] <= g["peak"] <= S.SPACE["peak"][1] and g["k"] == 1 for g in G))

st = S.run_study(None, None, both(), None, None, base, None, n_trials=40, storage=None, edit_mode="hook",
                 objective_mode="kl_first", grid=G)
first = [t.params for t in st.trials[:24]]
check("first 24 trials are the grid cells, in order", first == G)
check("sampler continues after the grid (40 trials, later ones differ from grid cells)",
      len(st.trials) == 40 and any(t.params not in G for t in st.trials[24:]))
fr = st.user_attrs.get("grid_frontier")
check("grid frontier recorded with every level", isinstance(fr, dict) and set(fr) == {"0.02", "0.05", "0.1", "0.2", "0.5"})
check("frontier is monotone: allowing more refusal never raises the best KL",
      all(fr[a] is None or fr[b] is None or fr[b][0] <= fr[a][0] + 1e-12 for a, b in zip(list(fr), list(fr)[1:])))
check("frontier(upto=24) equals the recorded grid frontier", S.frontier(st, upto=24) == fr)

one = S.run_study(None, None, {"plain": FakeSS()}, None, None, base, None, n_trials=14, storage=None, edit_mode="hook",
                  objective_mode="kl_first", grid=G)
check("cells of a missing subspace are dropped (12 plain cells run first)",
      [t.params for t in one.trials[:12]] == [g for g in G if g["subspace"] == "plain"])

with tempfile.TemporaryDirectory() as d:
    j = str(Path(d) / "study.log")
    a = S.run_study(None, None, both(), None, None, base, None, n_trials=10, study_name="g", storage=S.make_storage("journal", j),
                    edit_mode="hook", objective_mode="kl_first", grid=G)
    b = S.run_study(None, None, both(), None, None, base, None, n_trials=30, study_name="g", storage=S.make_storage("journal", j),
                    edit_mode="hook", objective_mode="kl_first", grid=G)
    done = [t for t in b.trials if t.values is not None]
    check("resume: 10 then 30 scored trials in total", len(done) == 30)
    check("resume: all 24 grid cells scored exactly once", sorted(map(str, [t.params for t in done if t.params in G])) == sorted(map(str, G)))

plain = S.run_study(None, None, both(), None, None, base, None, n_trials=30, storage=None, edit_mode="hook", objective_mode="kl_first", seed=3)
again = S.run_study(None, None, both(), None, None, base, None, n_trials=30, storage=None, edit_mode="hook", objective_mode="kl_first", seed=3)
check("without a grid nothing changed: same seed gives the same trials, no grid attr",
      [t.params for t in plain.trials] == [t.params for t in again.trials] and "grid_frontier" not in plain.user_attrs)

def pick(seed, grid):
    s = S.run_study(None, None, both(), None, None, base, None, n_trials=80, storage=None, edit_mode="hook",
                    objective_mode="kl_first", seed=seed, grid=grid)
    return S.best_plan(s, refusal_tol=0.02)["terms"]
rnd = [pick(s, None) for s in (0, 1, 2)]
grd = [pick(s, G) for s in (0, 1, 2)]
kr, kg = [t["kl"] for t in rnd], [t["kl"] for t in grd]
print("80 trials, selected KL by seed  random start:", [round(k, 4) for k in kr], " grid start:", [round(k, 4) for k in kg])
check("grid start: every seed ends inside the refusal tolerance", all(t["refusal_rate"] <= 0.02 for t in grd))
check("grid start: spread across seeds is no worse than the random start", max(kg) / min(kg) <= max(kr) / min(kr) + 1e-9)
check("grid start: worst seed is no worse than the random start's worst seed", max(kg) <= max(kr) + 1e-9)
print("ALL PASS" if ok else "FAILURES"); sys.exit(0 if ok else 1)
