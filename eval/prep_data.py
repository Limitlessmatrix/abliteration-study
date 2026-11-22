"""Held-out eval sets for the abliteration bake-off (stdlib only, CPU).

None of these sets is used for training by any contender. Every candidate prompt is filtered against
the prompts each contender trains on:
  ours        AdvBench, HarmBench (text_all), XSTest, Alpaca            (ablate/sources.py)
  Heretic     mlabonne/harmful_behaviors (AdvBench-derived), mlabonne/harmless_alpaca (Alpaca)
  OBLITERATUS built-in list in obliteratus/prompts.py (string literals read with ast, never imported),
              optional HarmBench/AdvBench loaders (covered above)
A candidate is dropped when it shares any 8-word shingle with a training prompt or its word-set
Jaccard with one is >= 0.5.

  harmful.jsonl      StrongREJECT (313) minus its AdvBench-sourced rows minus overlaps
  overrefusal.jsonl  300 from OR-Bench-hard-1k (seemingly toxic but benign), seed 0
  harmless.jsonl     300 Dolly-15k instructions without context (open_qa, general_qa, brainstorming,
                     creative_writing), seed 0; used for the KL reference
MANIFEST.json records source URLs, sha256 of every download and of every output, and drop counts.
usage: python -I prep_data.py OUT_DIR [OBLITERATUS_SRC_DIR]
OBLITERATUS_SRC_DIR is the source tree of OBLITERATUS v0.1.4 (commit 461ff086e675d93969bbd7c99c53505762452309);
its built-in prompt list is read as text (never imported) and used only to filter overlaps. Without it the
filter skips those prompts, so the sets can differ slightly from the study's (MANIFEST.json records which).
Only public dataset files are downloaded; nothing is bundled with this repository.
"""
import ast, csv, hashlib, io, json, random, re, sys, urllib.request
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(__doc__)
OUT = Path(sys.argv[1])
OBL = Path(sys.argv[2]) if len(sys.argv) > 2 else None
OUT.mkdir(parents=True, exist_ok=True)
HF = "https://huggingface.co/datasets"
SRC = {
    "strongreject": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
    "orbench_hard": f"{HF}/bench-llm/or-bench/resolve/e36d8b80e81837c8a8f264bbb2a49f1b32c7e272/or-bench-hard-1k.csv",
    "dolly": f"{HF}/databricks/databricks-dolly-15k/resolve/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl",
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "harmbench": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv",
    "alpaca": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
}
man = {"sources": {}, "outputs": {}, "dropped": {}}


def get(key):
    req = urllib.request.Request(SRC[key], headers={"User-Agent": "bakeoff-prep"})
    b = urllib.request.urlopen(req, timeout=120).read()
    man["sources"][key] = {"url": SRC[key], "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    return b.decode("utf-8")


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


WORD = re.compile(r"[a-z0-9]+")
def words(s): return WORD.findall(s.lower())


# ---- training prompts of all contenders
train = [r["goal"] for r in rows(get("advbench"))]
train += [r["Behavior"] for r in rows(get("harmbench"))]
train += [r["prompt"] for r in rows(get("xstest"))]
train += [(r["instruction"] + " " + r.get("input", "")).strip() for r in json.loads(get("alpaca"))]
obl_lits = [] if OBL is None else [
    n.value for n in ast.walk(ast.parse((OBL / "obliteratus" / "prompts.py").read_text(encoding="utf-8")))
    if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value.split()) >= 4]
train += obl_lits
man["train_refs"] = {"n": len(train), "obliteratus_builtin_literals": len(obl_lits),
                     "obliteratus_tree": str(OBL) if OBL else None}
SH = set()
ref_sets = []
for t in train:
    w = words(t)
    SH.update(" ".join(w[i:i + 8]) for i in range(len(w) - 7))
    if w:
        ref_sets.append(frozenset(w))
inv = {}
for i, s in enumerate(ref_sets):
    for w in s:
        inv.setdefault(w, []).append(i)


def contaminated(p):
    w = words(p)
    if any(" ".join(w[i:i + 8]) in SH for i in range(len(w) - 7)):
        return True
    s = frozenset(w)
    if not s:
        return True
    rare = sorted(s, key=lambda x: len(inv.get(x, ())))[:3]   # candidates share at least one rare word
    cand = {i for x in rare for i in inv.get(x, ())}
    return any(len(s & ref_sets[i]) / len(s | ref_sets[i]) >= 0.5 for i in cand)


def write(name, items):
    p = OUT / f"{name}.jsonl"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    man["outputs"][name] = {"n": len(items), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


sr = rows(get("strongreject"))
keep = [r for r in sr if r["source"] != "AdvBench"]
clean = [r for r in keep if not contaminated(r["forbidden_prompt"])]
man["dropped"]["harmful"] = {"advbench_source": len(sr) - len(keep), "overlap": len(keep) - len(clean)}
write("harmful", [{"id": f"sr{i}", "prompt": r["forbidden_prompt"], "category": r["category"]} for i, r in enumerate(clean)])

orb = rows(get("orbench_hard"))
cand = [r for r in orb if not contaminated(r["prompt"])]
man["dropped"]["overrefusal"] = {"overlap": len(orb) - len(cand)}
random.Random(0).shuffle(cand)
write("overrefusal", [{"id": f"or{i}", "prompt": r["prompt"], "category": r.get("category", "")} for i, r in enumerate(cand[:300])])

dolly = [json.loads(l) for l in get("dolly").splitlines() if l.strip()]
dolly = [d for d in dolly if d["category"] in ("open_qa", "general_qa", "brainstorming", "creative_writing")
         and not d["context"].strip() and 4 <= len(d["instruction"].split()) <= 80]
cand = [d for d in dolly if not contaminated(d["instruction"])]
man["dropped"]["harmless"] = {"pool": len(dolly), "overlap": len(dolly) - len(cand)}
random.Random(0).shuffle(cand)
write("harmless", [{"id": f"hl{i}", "prompt": d["instruction"], "category": d["category"]} for i, d in enumerate(cand[:300])])

(OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print(json.dumps({k: man[k] for k in ("outputs", "dropped", "train_refs")}, indent=1))
