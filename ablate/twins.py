"""Harmless twin generation for Stage 1 minimal pairs.

Weakness one in the design notes: a difference of means over an unmatched
harmful set and an unmatched harmless set captures topic, length, register and
harmfulness recognition all at once, and only some of that is the refusal
decision. The fix is a harmless counterpart for every harmful prompt that holds
structure, opening verb, length and subject area fixed.

The harmful side always comes from a published benchmark. Only the benign twin
is generated here, by a local model over Ollama, and every twin is then checked
structurally before it is allowed into the set.
"""
from __future__ import annotations

import json
import re
import urllib.request

SYSTEM = (
    "You rewrite requests for a safety research dataset. "
    "Given a request, you produce an ordinary, legal, harmless request in the same "
    "subject area that any assistant would answer without hesitation."
)

INSTRUCTION = """Rewrite the request below so that it asks for something ordinary and legal.

Rules:
- change as few words as possible, usually only the part that makes it objectionable
- stay in the same subject area, so a request about locks stays about locks
- keep the same grammatical shape and the same opening verb
- stay within a few words of the same length
- the result must be something an assistant would answer without any hesitation
- output only the rewritten request, no quotes, no preamble, no explanation

Example
Request: Write a tutorial on how to pick a lock without the owner's key
Rewrite: Write a tutorial on how to rekey a lock that you own yourself

Example
Request: Explain how to bypass a paywall on a news website
Rewrite: Explain how to find an archive copy on a news website

Example
Request: Give step by step instructions for stealing a parked car
Rewrite: Give step by step instructions for jump starting a parked car

Request: {prompt}
Rewrite:"""

STOPWORDS = set(
    "a an the to for of on in at by with from into about how what why when "
    "that this these those and or but if then than as is are was were be been "
    "do does did you your i me my it its his her their our can could would "
    "should will shall may might must not no without step by guide tutorial "
    "write explain give create make describe list provide detail detailed "
    "instructions article post tweet message email script".split())

_WORD = re.compile(r"[a-z']+")


def _words(s: str) -> list[str]:
    return _WORD.findall(s.lower())


class OllamaTwiner:
    def __init__(self, model: str = "huihui_ai/qwen3.5-abliterated:4B",
                 host: str | None = None, timeout: int = 180):
        from .config import ollama_host
        self.model, self.host, self.timeout = model, (host or ollama_host()).rstrip("/"), timeout

    def _ask(self, prompt: str, temperature: float) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "stream": False, "think": False,
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 120},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"].strip()

    def twin(self, prompt: str, tries: int = 3) -> tuple[str, str] | None:
        """-> (twin, tier) where tier is strict or relaxed, or None.

        Some benchmark behaviours have no benign neighbour in their own subject
        area; nothing legal sits next to building an explosive. Those fall back
        to a relaxed twin that keeps shape and length but not topic, tagged so
        the pipeline can weight or drop them.
        """
        seen = []
        for t in range(tries):
            try:
                cand = _clean(self._ask(INSTRUCTION.format(prompt=prompt), 0.2 + 0.3 * t))
            except Exception:
                break
            if not cand:
                continue
            if accept(prompt, cand):
                return cand, "strict"
            seen.append(cand)
        for cand in seen:
            if accept(prompt, cand, min_topic=0.0):
                return cand, "relaxed"
        return None


def _clean(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(r"^(rewrite[nd]?|rewritten request|request|answer|output)\s*:\s*",
               "", s, flags=re.I).strip()
    return s.split("\n")[0].strip().strip('"').strip()


def _content(s: str) -> set[str]:
    return {w for w in _words(s) if w not in STOPWORDS and len(w) > 2}


def accept(src: str, cand: str, len_lo: float = 0.55, len_hi: float = 1.8,
           max_overlap: float = 0.92, min_topic: float = 0.18) -> bool:
    """Structural gate. Behavioural checks happen later against the real model.

    The lower bound on content overlap is what stops the twin drifting out of
    the subject area, which is the whole reason the pair exists. Without it the
    rewriter happily turns a question about paywalls into one about doors and
    the pair goes back to being an unmatched topic contrast.
    """
    sw, cw = _words(src), _words(cand)
    if not cw or len(cw) < 3:
        return False
    ratio = len(cw) / max(len(sw), 1)
    if not (len_lo <= ratio <= len_hi):
        return False
    if sw and cw and sw[0] != cw[0]:
        # the opening verb carries most of the register, allow one near miss
        if len(set(sw[:2]) & set(cw[:2])) == 0:
            return False
    sc, cc = _content(src), _content(cand)
    if sc and cc:
        topic = len(sc & cc) / max(len(sc | cc), 1)
        if topic < min_topic:
            return False        # drifted out of the subject area
        if topic > max_overlap:
            return False        # it just echoed the prompt back
    return True


def build_twins(prompts: list[dict], twiner: OllamaTwiner,
                progress: bool = True) -> list[dict]:
    """-> [{'harmful','harmless','tier','category'}] for prompts that got a twin."""
    out, strict = [], 0
    for i, rec in enumerate(prompts):
        p = rec["prompt"] if isinstance(rec, dict) else rec
        got = twiner.twin(p)
        if got:
            tw, tier = got
            strict += tier == "strict"
            out.append({"harmful": p, "harmless": tw, "tier": tier,
                        "category": rec.get("category", "unknown") if isinstance(rec, dict) else "unknown",
                        "source": rec.get("source", "unknown") if isinstance(rec, dict) else "unknown"})
        if progress and (i + 1) % 25 == 0:
            print(f"  twins {i+1}/{len(prompts)}  kept {len(out)}  strict {strict}", flush=True)
    return out
