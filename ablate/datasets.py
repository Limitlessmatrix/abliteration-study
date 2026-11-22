"""Stage 1: assemble the working sets with strict splits.

Five sets come out of here:

  pairs        harmful behaviour from a public benchmark plus its generated
               harmless twin, split train/val/test
  corpus       broad harmless instructions, the KL measurement set and the
               Stage 5 healing corpus
  borderline   XSTest safe prompts, the over refusal probe
  calibration  the four protected behaviours with matched controls
  probe        a behavioural labelling of every prompt by the target model,
               which is what lets the refusal decision be separated from
               harmfulness recognition

Splits are assigned by a stable hash of the prompt text, so adding sources
later does not reshuffle what was already held out.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from . import calibration, sources
from .twins import OllamaTwiner, build_twins

SPLITS = ("train", "val", "test")


def split_of(text: str, weights=(0.6, 0.2, 0.2), salt: str = "ablate") -> str:
    h = hashlib.sha256((salt + "|" + text).encode()).digest()
    x = int.from_bytes(h[:8], "big") / 2 ** 64
    c = 0.0
    for name, w in zip(SPLITS, weights):
        c += w
        if x < c:
            return name
    return SPLITS[-1]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build(root: Path, harmful_source: str = "advbench", n_harmful: int = 300,
          n_corpus: int = 600, seed: int = 0, twiner_model: str | None = None,
          skip_twins: bool = False) -> dict:
    """Fetch, twin, split and write. Returns a summary dict."""
    root = Path(root)
    sets_dir = root / "sets"
    sources.fetch(root)

    rng = random.Random(seed)

    harmful = sources.load_harmful(root, harmful_source)
    rng.shuffle(harmful)
    harmful = harmful[:n_harmful]

    twin_path = sets_dir / "pairs_raw.jsonl"
    if skip_twins and twin_path.exists():
        pairs = read_jsonl(twin_path)
    else:
        tw = OllamaTwiner(twiner_model) if twiner_model else OllamaTwiner()
        print(f"generating harmless twins for {len(harmful)} behaviours", flush=True)
        pairs = build_twins(harmful, tw)
        write_jsonl(twin_path, pairs)

    for p in pairs:
        p["split"] = split_of(p["harmful"])
    write_jsonl(sets_dir / "pairs.jsonl", pairs)

    corpus = sources.load_alpaca(root, max_n=None)
    rng.shuffle(corpus)
    corpus = corpus[:n_corpus]
    for c in corpus:
        c["split"] = split_of(c["prompt"], salt="corpus")
    write_jsonl(sets_dir / "corpus.jsonl", corpus)

    safe, unsafe = sources.load_xstest(root)
    for s in safe:
        s["split"] = split_of(s["prompt"], salt="xstest")
    write_jsonl(sets_dir / "borderline.jsonl", safe)
    write_jsonl(sets_dir / "xstest_unsafe.jsonl", unsafe)

    cal = calibration.items()
    for c in cal:
        c["split"] = split_of(c["probe"], salt="cal")
    write_jsonl(sets_dir / "calibration.jsonl", cal)

    summary = {
        "harmful_source": harmful_source,
        "pairs": len(pairs),
        "pairs_strict": sum(p.get("tier") == "strict" for p in pairs),
        "pairs_by_split": {s: sum(p["split"] == s for p in pairs) for s in SPLITS},
        "corpus": len(corpus),
        "borderline": len(safe),
        "calibration": len(cal),
        "calibration_by_category": {
            c: sum(1 for x in cal if x["category"] == c) for c in calibration.CATEGORIES},
    }
    (sets_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    return summary


def load(root: Path, split: str | None = None) -> dict:
    """Read the built sets back, optionally filtered to one split."""
    sets_dir = Path(root) / "sets"
    out = {}
    for name in ("pairs", "corpus", "borderline", "calibration"):
        p = sets_dir / f"{name}.jsonl"
        rows = read_jsonl(p) if p.exists() else []
        if split:
            rows = [r for r in rows if r.get("split") == split]
        out[name] = rows
    return out


def label_behaviour(model, tok, rows: list[dict], key: str, device: str = "cuda",
                    max_new_tokens: int = 48, batch_size: int = 16) -> list[dict]:
    """Tag each row with whether the target model actually refuses it.

    This is the gate that makes a pair real. A harmful prompt the model answers
    anyway and a twin the model refuses both contribute noise to the difference,
    and the four way labelling is what Stage 2.4 needs to tell the refusal
    decision apart from harmfulness recognition.
    """
    from .capture import generate
    from .refusal import is_refusal

    prompts = [r[key] for r in rows]
    outs = generate(model, tok, prompts, max_new_tokens=max_new_tokens,
                    batch_size=batch_size, device=device)
    for r, o in zip(rows, outs):
        r[f"{key}_refused"] = bool(is_refusal(o))
        r[f"{key}_response"] = o[:200]
    return rows
