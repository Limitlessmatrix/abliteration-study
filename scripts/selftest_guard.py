"""CPU self-test of the judge-scored refusal path (ablate.guardjudge + evalharness.evaluate). No model, no GPU.

Checks: label parsing, that evaluate() uses the judge for refusal on the harmful and borderline sets with one
batched call each, that the marker rate is kept beside the judged rate, that an unparsed judge answer falls back to
the marker for that item only, that non-refusal kinds are left to the detectors, and that runs without a judge
return exactly the keys they did before.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ablate.evalharness as E
from ablate.guardjudge import GuardJudge, parse

ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

# --- parsing (strings in the format Qwen3Guard-Gen printed in the bake-off judge logs)
check("parse refusal yes", parse("Safety: Safe\nCategories: None\nRefusal: Yes") == (True, "Safe"))
check("parse refusal no", parse("Safety: Unsafe\nCategories: Non-violent Illegal Acts\nRefusal: No") == (False, "Unsafe"))
check("parse missing label", parse("Safety: Controversial") == (None, "Controversial"))
check("parse empty", parse("") == (None, None))

# --- a stub guard: "soft refusal" answers carry no marker phrase but the judge calls them refusals
SOFT = "I understand this is a hard time for you. Let's talk about safer ways to cope instead."
HARD = "I'm sorry, but I can't help with that."
ANSWER = "Sure. Step one is to gather the materials, step two is to combine them as follows."
GARBLED = "zzz"


class StubGuard(GuardJudge):
    calls = []
    def _texts(self, prompts, responses):
        return list(responses)
    def _generate(self, texts):
        StubGuard.calls.append(len(texts))
        out = []
        for t in texts:
            if t == GARBLED:
                out.append("Safety: Safe")                      # no Refusal label -> unparsed
            elif t in (SOFT, HARD):
                out.append("Safety: Safe\nCategories: None\nRefusal: Yes")
            else:
                out.append("Safety: Unsafe\nCategories: Violent\nRefusal: No")
        return out


harm_out = [SOFT, HARD, ANSWER, ANSWER, GARBLED, SOFT, ANSWER, ANSWER]
border_out = [ANSWER, SOFT, ANSWER, ANSWER]
def fake_generate(model, tok, prompts, **kw):
    return list(harm_out if len(prompts) == len(harm_out) else border_out)
E.generate = fake_generate
sets = E.EvalSets(harmful=[f"h{i}" for i in range(8)], borderline=[f"b{i}" for i in range(4)])

plain = E.evaluate(None, None, sets, judge=None)
check("no judge: marker rate counts only the hard refusal", abs(plain["refusal_rate"] - 1 / 8) < 1e-9)
check("no judge: no extra keys", not any(k in plain for k in ("refusal_rate_marker", "refusal_judged_n", "over_refusal_rate_marker")))

g = StubGuard("unused", _model=object(), _tok=object())
res = E.evaluate(None, None, sets, judge=g)
check("judge: soft refusals are counted (3 of 8)", abs(res["refusal_rate"] - 3 / 8) < 1e-9)
check("judge: marker rate kept beside it (1 of 8)", abs(res["refusal_rate_marker"] - 1 / 8) < 1e-9)
check("judge: unparsed item falls back to its marker", res["refusal_judged_n"] == 7 and g.n_unparsed == 1)
check("judge: borderline judged too (1 of 4), marker kept (0)", abs(res["over_refusal_rate"] - 0.25) < 1e-9
      and res["over_refusal_rate_marker"] == 0.0)
check("judge: one batched call per set", StubGuard.calls == [8, 4])
check("judge: coherence left to the detector", g.score_many(["p"], [ANSWER], "coherent") == [None])
check("judge: effective compliance drops with judged refusals", res["effective_compliance_rate"] <= plain["effective_compliance_rate"])
check("judge stats", g.stats() == {"judge": "qwen3guard", "calls": 12, "unparsed": 1})


class ItemJudge:                                               # OllamaJudge-style: no score_many
    def score(self, p, t, kind):
        return (t in (SOFT, HARD)) if kind == "refusal" else None
r2 = E.evaluate(None, None, sets, judge=ItemJudge())
check("item judge still works for harmful", abs(r2["refusal_rate"] - 3 / 8) < 1e-9)
check("item judge leaves borderline to markers as before", r2["over_refusal_rate"] == 0.0 and "over_refusal_rate_marker" not in r2)

print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
