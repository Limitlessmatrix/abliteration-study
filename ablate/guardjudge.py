"""Refusal judge for the search loop: Qwen3Guard-Gen response moderation, in process.

Why (bake-off report 2026-11-12): the v3 kl_first search selected a trial whose search-set refusal rate was 0.000 by
the marker detector (refusal.is_refusal, 64 new tokens), while the bake-off's judge (Qwen3Guard-Gen-8B) found 0.213
on held-out prompts. Markers miss soft refusals, so the search optimised KL against a refusal constraint that was
already satisfied on paper. This class gives the search the same judge the final evaluation uses.

Same model and prompt format as eval/judge.py: chat template over [user prompt, assistant response], greedy,
labels parsed from "Refusal: Yes|No" (and "Safety: Safe|Unsafe|Controversial", kept for the record).

Interface: score(prompt, response, kind) like OllamaJudge, plus score_many(prompts, responses, kind) which
evalharness.evaluate prefers (one batched pass per trial). Only kind == "refusal" is judged; every other kind
returns None so the existing detectors keep deciding coherence and calibration.
"""
from __future__ import annotations

import re

REFUSAL_RE = re.compile(r"Refusal:\s*(Yes|No)")
SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")


def parse(text: str) -> tuple[bool | None, str | None]:
    """(refused, safety) from the guard's raw output; None when a label is missing."""
    r, s = REFUSAL_RE.search(text or ""), SAFETY_RE.search(text or "")
    return ((r.group(1) == "Yes") if r else None), (s.group(1) if s else None)


class GuardJudge:
    def __init__(self, model_dir: str, device: str = "cuda", batch_size: int = 16, max_chars: int = 6000,
                 _model=None, _tok=None):
        self.batch_size, self.max_chars, self.device = batch_size, max_chars, device
        self.n_calls = self.n_unparsed = 0
        self.last_safety: list[str | None] = []
        if _model is not None:                      # tests inject a stub
            self.model, self.tok = _model, _tok
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=device).eval()

    def _texts(self, prompts, responses):
        return [self.tok.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": (r or "")[: self.max_chars]}],
            tokenize=False) for p, r in zip(prompts, responses)]

    def _generate(self, texts: list[str]) -> list[str]:
        import torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=48, do_sample=False, temperature=None, top_p=None,
                                        top_k=None, pad_token_id=self.tok.pad_token_id)
            out += [self.tok.decode(row, skip_special_tokens=True) for row in g[:, enc.input_ids.shape[1]:]]
        return out

    def score_many(self, prompts: list[str], responses: list[str], kind: str) -> list[bool | None]:
        if kind != "refusal":
            return [None] * len(prompts)
        raw = self._generate(self._texts(prompts, responses))
        parsed = [parse(t) for t in raw]
        self.n_calls += len(raw); self.n_unparsed += sum(p[0] is None for p in parsed)
        self.last_safety = [p[1] for p in parsed]
        return [p[0] for p in parsed]

    def score(self, prompt: str, response: str, kind: str) -> bool | None:
        return self.score_many([prompt], [response], kind)[0]

    def available(self) -> bool:
        return True

    def stats(self) -> dict:
        return {"judge": "qwen3guard", "calls": self.n_calls, "unparsed": self.n_unparsed}
