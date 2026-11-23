"""Parse the result tables of PAPER.md section 5 into results.json / results.csv (one row per model x contender x seed).
Run: python tools/tables_to_json.py paper/PAPER.md paper/data"""
import csv, json, re, sys
src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
sec = text.split("## 5. Results", 1)[1].split("## 6.", 1)[0]
MODEL = {"5.1": "Qwen3-8B", "5.2": "gemma-3-12b-it", "5.3": "Qwen3-32B", "5.6": "GLM-5.3-Flash (Maya-XS IQ1_S)"}
COLS = {"refusal": "refusal", "judged refusal": "refusal", "answered unsafe": "harm_comply", "over-refusal": "overrefusal",
        "kl": "kl", "first-token kl": "kl_first", "mmlu": "mmlu", "arc": "arc", "gsm8k": "gsm8k",
        "phrase-count refusal": "refusal_phrase"}
rows = []
for part in re.split(r"\n### ", sec)[1:]:
    num = part.split(" ", 1)[0]
    lines = [l for l in part.splitlines() if l.startswith("|")]
    if len(lines) < 3:
        continue
    head = [h.strip().lower() for h in lines[0].strip("|").split("|")]
    for l in lines[2:]:
        cells = [c.strip().replace("**", "") for c in l.strip("|").split("|")]
        name = cells[0]
        model = MODEL.get(num)
        if num == "5.4":
            model, name = ("Qwen3.8-27B", name.replace("Qwen3.8-27B ", "")) if name.startswith("Qwen3.8-27B") else \
                          ("gemma-3-27b-it", name.replace("gemma-3-27b-it ", "")) if name.startswith("gemma-3-27b-it") else (rows[-1]["model"], name)
        m = re.search(r"\(seeds ([\d / ]+)\)", name)
        seeds = [s.strip() for s in m.group(1).split("/")] if m else [None]
        base = re.sub(r"\s*\(seeds [\d / ]+\)", "", name)
        for i, sd in enumerate(seeds):
            r = {"model": model, "contender": base, "seed": int(sd) if sd else None,
                 "split": "search" if num == "5.5" else "heldout", "section": num}
            for h, c in zip(head[1:], cells[1:]):
                k = COLS.get(h)
                if not k or not c:
                    continue
                v = c.split("/")[i].strip() if "/" in c else c
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
            rows.append(r)
json.dump({"paper": "Removing refusals with less damage (draft 3, 2026-11-22)", "metrics": {
    "refusal": "share of 282 held-out StrongREJECT prompts the Qwen3Guard-Gen-8B judge marks as refused",
    "harm_comply": "share answered and judged unsafe", "overrefusal": "share of 300 OR-Bench-hard prompts refused",
    "kl": "mean per-token KL from the base on its own answers to 300 Dolly prompts (100 for GLM)",
    "kl_first": "first-token KL", "mmlu": "accuracy", "arc": "ARC-Challenge accuracy", "gsm8k": "accuracy",
    "refusal_phrase": "phrase-matched refusal rate (shown only to compare with the judge)"},
    "rows": rows}, open(f"{out}/results.json", "w"), indent=1)
keys = ["model", "contender", "seed", "split", "section", "refusal", "refusal_phrase", "harm_comply", "overrefusal", "kl", "kl_first", "mmlu", "arc", "gsm8k"]
with open(f"{out}/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k) for k in keys}) for r in rows]
print(len(rows), "rows")
