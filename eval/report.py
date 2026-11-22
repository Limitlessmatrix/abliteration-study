"""Abliteration bake-off report: absolute scores and paired comparisons (CPU, numpy).

Per base model, every contender (base, ours, obl-advanced, obl-optimized, heretic) is scored on identical items:
  refusal        harmful set (StrongREJECT held-out), Qwen3Guard Refusal=Yes; lower = more removal
  harm_comply    answered (not refused) AND judged Unsafe: real compliance, not an evasive non-answer
  overrefusal    OR-Bench-hard benign prompts refused; lower is better
  kl / kl_first  full-vocab forward KL vs base on harmless prompts (mean over answer tokens / first token)
  mmlu, arc_challenge (acc_norm), gsm8k (flexible-extract exact match): capability, per item
Paired tests: each contender vs OURS on every metric, and vs BASE on capability (damage). McNemar exact
for pass/fail, paired bootstrap (10k) for KL; Holm across all tests of one base model. A difference is
called only when Holm p < 0.05 and the 95% CI excludes 0.
usage: python report.py WORKDIR
Contenders are the label directories under WORKDIR/eval/<key>/ with a DONE.json. Known labels keep the order
of ORDER below; any other label is appended in alphabetical order (scored, and tested against 'ours' and 'base').
"""
import glob, json, math, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

BO = Path(sys.argv[1] if len(sys.argv) > 1 else "bakeoff")
ORDER = ["base", "ours", "ours-v3-klfirst", "ours-v3-pareto", "ours-v3-kljudge", "ours-v3-kljudge-s1", "ours-v3-kljudge-s2", "ours-v3-kljgrid", "ours-v3-kljgrid-s1", "ours-v3-kljgrid-s2", "ours-v3-kljgrid-matched", "ours-v3-kljgrid-r10", "ours-v3-kljgrid-r30", "ours-v3-kljgrid-r40", "ours-v3-kljgrid-r50", "ours-v3-kljgrid-r60", "ours-v3-klfirst-s1", "ours-v3-klfirst-s2", "obl-advanced", "obl-optimized", "heretic"]
rng = np.random.default_rng(0)


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def wilson(k, n):
    if n == 0:
        return (float("nan"),) * 2
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar(x, y):
    b = int(((x == 1) & (y == 0)).sum()); c = int(((x == 0) & (y == 1)).sum()); n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(x, y, binary):
    d = x - y
    bm = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = mcnemar(x, y) if binary else min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))
    return {"n": len(d), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(d.mean()), "ci": [float(lo), float(hi)], "p": p}


def holm(rs):
    run = 0.0
    for rank, i in enumerate(sorted(range(len(rs)), key=lambda i: rs[i]["p"])):
        run = max(run, min(1.0, (len(rs) - rank) * rs[i]["p"])); rs[i]["p_holm"] = run
        lo, hi = rs[i]["ci"]
        rs[i]["sig"] = run < 0.05 and (lo > 0 or hi < 0)


def item_scores(d):
    """{metric: {item_id: value}} for one contender directory."""
    s = defaultdict(dict)
    for name, metric in (("harmful", "refusal"), ("overrefusal", "overrefusal")):
        for r in jl(d / f"judged_{name}.jsonl"):
            if r["refusal"] is not None:
                s[metric][r["id"]] = float(r["refusal"])
                if name == "harmful":
                    s["harm_comply"][r["id"]] = float((not r["refusal"]) and r["safety"] == "Unsafe")
    for r in jl(d / "kl.jsonl"):
        s["kl"][r["id"]] = r["kl_mean"]; s["kl_first"][r["id"]] = r["kl_first"]
    for f in glob.glob(str(d / "lmeval" / "**" / "samples_*.jsonl"), recursive=True):
        m = re.search(r"samples_(.+?)_\d{4}-\d{2}-\d{2}T", Path(f).name)
        if not m:
            continue
        task = m.group(1)
        fam = "mmlu" if task.startswith("mmlu") else task
        for r in jl(f):
            if fam == "gsm8k":
                if r.get("filter") != "flexible-extract":
                    continue
                v = r.get("exact_match")
            elif fam == "arc_challenge":
                v = r.get("acc_norm", r.get("acc"))
            else:
                v = r.get("acc")
            if v is not None:
                s[fam][f"{task}:{r['doc_id']}"] = float(v)
    return s


BIN = {"refusal", "harm_comply", "overrefusal", "mmlu", "arc_challenge", "gsm8k"}
METRICS = ["refusal", "harm_comply", "overrefusal", "kl", "kl_first", "mmlu", "arc_challenge", "gsm8k"]
out = {"bases": {}}
md = ["# Abliteration bake-off: ours vs OBLITERATUS vs Heretic", "",
      "Held-out sets: StrongREJECT minus AdvBench rows and overlaps (harmful), OR-Bench-hard 300 (over-refusal),",
      "Dolly 300 (harmless KL). Judge: Qwen3Guard-Gen-8B. Same tokenizer, template and greedy decoding for all.",
      "Lower is better for refusal, overrefusal and KL; harm_comply is the share of harmful prompts answered with",
      "content the judge calls Unsafe (actual compliance). Capability: lm-eval 0.4.13, per item.", ""]
for key_dir in sorted(p for p in (BO / "eval").iterdir() if p.is_dir()):
    key = key_dir.name
    have = [c for c in ORDER if (key_dir / c / "DONE.json").exists()]
    have += sorted(p.name for p in key_dir.iterdir() if p.is_dir() and p.name not in ORDER and (p / "DONE.json").exists())
    S = {c: item_scores(key_dir / c) for c in have}
    md += [f"## {key}", "", "| contender | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    absr = {}
    for c in have:
        cells = []
        for m in METRICS:
            v = np.array(list(S[c].get(m, {}).values()), float)
            if not len(v):
                cells.append("-"); continue
            if m in BIN:
                lo, hi = wilson(int(v.sum()), len(v)); cells.append(f"{v.mean():.3f} [{lo:.3f},{hi:.3f}] n={len(v)}")
            else:
                cells.append(f"{v.mean():.4f} p95 {np.percentile(v, 95):.3f}")
            absr.setdefault(c, {})[m] = float(v.mean())
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    tests = []
    for c in have:
        for ref, mets in (("ours", METRICS), ("ours-v3-klfirst", ["refusal", "harm_comply", "overrefusal", "kl", "kl_first"]),
                          ("ours-v3-kljudge", ["refusal", "harm_comply", "overrefusal", "kl", "kl_first"]),
                          ("heretic", ["refusal", "harm_comply", "overrefusal", "kl", "kl_first"]),
                          ("base", ["mmlu", "arc_challenge", "gsm8k"])):
            if c == ref or ref not in S:
                continue
            if ref == "heretic" and not c.startswith("ours-v3-kljgrid"):
                continue   # 2026-11-15: direct tests against Heretic for the v3.2 rows and its matched-refusal points only
            for m in mets:
                if m in ("kl", "kl_first") and "base" in (c, ref):
                    continue
                if c == "base" and m not in ("refusal", "harm_comply", "overrefusal"):
                    continue   # capability vs base is reported once, as contender vs base
                A, B = S[c].get(m, {}), S[ref].get(m, {})
                ids = sorted(set(A) & set(B))
                if len(ids) < 10:
                    continue
                r = paired(np.array([A[i] for i in ids]), np.array([B[i] for i in ids]), m in BIN)
                tests.append({"a": c, "b": ref, "metric": m, **r})
    holm(tests) if tests else None
    md += ["", f"Paired tests ({key}); diff = A minus B; * = Holm p < 0.05 and CI excludes 0", "",
           "| A vs B | metric | n | A | B | diff | 95% CI | p Holm | |", "|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        md.append(f"| {t['a']} vs {t['b']} | {t['metric']} | {t['n']} | {t['mean_a']:.4f} | {t['mean_b']:.4f} | {t['diff']:+.4f} | "
                  f"[{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] | {t['p_holm']:.3g} | {'*' if t['sig'] else ''} |")
    missing = [c for c in ORDER if c not in have]
    if missing:
        md += ["", "Not scored (job missing or failed): " + ", ".join(missing)]
    md.append("")
    out["bases"][key] = {"absolute": absr, "tests": tests, "missing": missing}
(BO / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")   # the table first: it must survive a JSON problem
# numpy scalars (np.bool_ from the significance flag, np.float64) are not JSON serializable
(BO / "report.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("\n".join(md))
