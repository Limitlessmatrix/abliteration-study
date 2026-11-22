"""Canonical public prompt sets.

The harmful side of every contrastive pair comes from a published refusal
benchmark so that refusal rates here are comparable with Arditi et al. 2024,
Heretic, and the HarmBench/JailbreakBench leaderboards. Nothing harmful is
authored in this repo.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

UA = "ablate-research/0.1"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    fname: str
    note: str


SOURCES = [
    Source("advbench", "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
           "advbench_harmful_behaviors.csv", "Zou et al. 2023, 520 behaviours, columns goal,target"),
    Source("harmbench", "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
           "harmbench_behaviors_text_all.csv", "Mazeika et al. 2024, column Behavior plus SemanticCategory"),
    Source("xstest", "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv",
           "xstest_prompts.csv", "Rottger et al. 2024, exaggerated safety, 450 prompts labelled safe or unsafe"),
    Source("alpaca", "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
           "alpaca_data.json", "Taori et al. 2023, 52k harmless instructions"),
]

BY_NAME = {s.name: s for s in SOURCES}


def fetch(root: Path, only: list[str] | None = None, force: bool = False) -> dict[str, Path]:
    """Download the public sets into root/public. Returns name -> path."""
    out_dir = root / "public"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    paths = {}
    for s in SOURCES:
        if only and s.name not in only:
            continue
        dest = out_dir / s.fname
        if dest.exists() and not force:
            paths[s.name] = dest
            continue
        req = urllib.request.Request(s.url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
        dest.write_bytes(blob)
        manifest[s.name] = {
            "url": s.url,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "note": s.note,
        }
        paths[s.name] = dest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return paths


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_harmful(root: Path, which: str = "advbench") -> list[dict]:
    """Harmful behaviours as [{'prompt','category','source'}]."""
    p = root / "public" / BY_NAME[which].fname
    if which == "advbench":
        return [{"prompt": r["goal"].strip(), "category": "advbench", "source": "advbench"}
                for r in _rows(p) if r.get("goal", "").strip()]
    if which == "harmbench":
        return [{"prompt": r["Behavior"].strip(),
                 "category": (r.get("SemanticCategory") or "unknown").strip(),
                 "source": "harmbench"}
                for r in _rows(p)
                if r.get("Behavior", "").strip() and r.get("FunctionalCategory", "standard") == "standard"]
    raise ValueError(which)


def load_xstest(root: Path) -> tuple[list[dict], list[dict]]:
    """(safe, unsafe) XSTest prompts. The safe ones are the over refusal probes:
    benign requests whose surface form reads as harmful, such as asking how to
    kill a process."""
    p = root / "public" / BY_NAME["xstest"].fname
    safe, unsafe = [], []
    for r in _rows(p):
        rec = {"prompt": (r.get("prompt") or "").strip(),
               "category": (r.get("type") or "").strip(),
               "focus": (r.get("focus") or "").strip(),
               "source": "xstest"}
        if not rec["prompt"]:
            continue
        (safe if (r.get("label") or "").strip().lower() == "safe" else unsafe).append(rec)
    return safe, unsafe


def load_alpaca(root: Path, max_n: int | None = None, no_input_only: bool = True) -> list[dict]:
    p = root / "public" / BY_NAME["alpaca"].fname
    data = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for r in data:
        if no_input_only and r.get("input"):
            continue
        out.append({"prompt": r["instruction"].strip(), "category": "alpaca", "source": "alpaca"})
        if max_n and len(out) >= max_n:
            break
    return out
