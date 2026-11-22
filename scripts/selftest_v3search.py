"""CPU self-test of the v3 (pareto) search: synthetic evaluate(), no model. Checks that the two objective
study runs, that the pick is the lowest KL trial inside tolerance, that scalar mode still works, and that a
journal study resumes instead of restarting."""
import sys, tempfile, types
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

def fake_eval(model, tok, sets, **kw):
    k, plan = FakeHook.cur
    strength = plan.peak * min(1.0, plan.width / 12) * (1.0 if plan.kinds == ("attn_out", "mlp_out") else 0.6)
    refusal = max(0.0, 0.95 - 1.6 * strength)          # stronger edit -> fewer refusals
    kl = 0.002 + 0.09 * strength ** 2 * (1 + 0.5 * (k - 1))   # and more damage, worse at high rank
    return {"refusal_rate": refusal, "kl_mean": kl, "coherence_rate": 1.0, "over_refusal_rate": 0.0, "calibration": {}}

S.HookEdit, S.evaluate = FakeHook, fake_eval
S.objective = lambda b, r, lambda_kl=1.0, lambda_cal=2.0: {"value": r["refusal_rate"] + r["kl_mean"], "terms": {
    "refusal_rate": r["refusal_rate"], "kl": r["kl_mean"], "calibration_drop": 0.0, "over_refusal_delta": 0.0, "coherence_drop": 0.0}}
import ablate.modelio as M
M.n_layers = lambda m: 36; M.has_block_writers = lambda m: False
base = {"refusal_rate": 0.95}
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

sc = S.run_study(None, None, {"protected": FakeSS(), "plain": FakeSS()}, None, None, base, None, n_trials=60,
                 storage=None, edit_mode="hook", objective_mode="scalar")
bs = S.best_plan(sc)
pa = S.run_study(None, None, {"protected": FakeSS(), "plain": FakeSS()}, None, None, base, None, n_trials=60,
                 storage=None, edit_mode="hook", objective_mode="pareto", kl_target=0.01)
bp = S.best_plan(pa, refusal_tol=0.02)
print("scalar pick ", bs["terms"], "| pareto pick", bp["terms"], bp["selected_by"])
check("scalar mode still returns a best trial", bs["terms"]["kl"] is not None)
check("pareto study has two directions", len(pa.directions) == 2)
check("pareto pick is inside refusal tolerance", bp["terms"]["refusal_rate"] <= 0.02)
kf = S.run_study(None, None, {"protected": FakeSS(), "plain": FakeSS()}, None, None, base, None, n_trials=60,
                 storage=None, edit_mode="hook", objective_mode="kl_first")
bk = S.best_plan(kf, refusal_tol=0.02)
print("kl_first pick", bk["terms"], bk["selected_by"], "| true optimum KL 0.0338 at refusal 0")
check("kl_first pick is inside refusal tolerance", bk["terms"]["refusal_rate"] <= 0.02)
check("kl_first KL <= pareto KL (same 60 trials)", bk["terms"]["kl"] <= bp["terms"]["kl"] + 1e-9)
check("kl_first within 15% of the true optimum", bk["terms"]["kl"] <= 0.0338 * 1.15)
inside = [t for t in pa.trials if t.user_attrs["refusal_rate"] <= 0.02]
check("pick is the minimum KL among qualifying trials", abs(min(t.user_attrs["kl"] for t in inside) - bp["terms"]["kl"]) < 1e-12)
check("pareto() front is non-empty and sorted out", len(S.pareto(pa)) >= 1)
with tempfile.TemporaryDirectory() as d:
    j = str(Path(d) / "study.log")
    a = S.run_study(None, None, {"protected": FakeSS()}, None, None, base, None, n_trials=12, study_name="r",
                    storage=S.make_storage("journal", j), edit_mode="hook", objective_mode="pareto")
    b = S.run_study(None, None, {"protected": FakeSS()}, None, None, base, None, n_trials=20, study_name="r",
                    storage=S.make_storage("journal", j), edit_mode="hook", objective_mode="pareto")
    check("journal resume: 12 then 20 total, not 32", len(b.trials) == 20)
tight = S.best_plan(pa, refusal_tol=-1.0)
check("no qualifying trial falls back to best scalar", "no trial inside tolerance" in tight["selected_by"])
print("ALL PASS" if ok else "FAILURES"); sys.exit(0 if ok else 1)
