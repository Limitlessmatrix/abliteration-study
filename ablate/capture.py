"""Stage 2.0 activation capture, plus the generation and teacher logit caching
the eval harness needs.

Residuals are taken at the last prompt token with left padding, which is the
position the refusal decision is made at for a chat formatted single turn.
"""
from __future__ import annotations

import torch

from .modelio import batched, format_chat, input_device, n_suffix_tokens


def magnitude_clip(x: torch.Tensor, percentile: float = 0.995) -> torch.Tensor:
    """Symmetric Winsorisation of activation components.

    Gemma style models carry a handful of outlier channels two orders of
    magnitude above the rest, and those channels dominate any mean or dot
    product taken over the raw activation. Grimjim reports needing
    sparsification at 0.995 on Gemma because the outliers obscured the refusal
    direction outright. We measured the same pathology from the other end: an
    uncentred gate projection put the harmless threshold at +827 against a
    harmful mean of -136.
    """
    if percentile >= 1.0:
        return x
    d = x.double()
    thr = torch.quantile(d.abs().flatten(), percentile)
    return torch.clamp(d, min=-thr, max=thr).to(x.dtype)


@torch.no_grad()
def capture_residuals(model, tok, prompts: list[str], system: str | None = None,
                      batch_size: int = 16, max_len: int = 512,
                      device: str = "cuda", positions: str = "post_instruction",
                      clip: float = 1.0) -> torch.Tensor:
    """-> float32 [n_prompts, n_layers+1, hidden] on CPU.

    Index 0 on the layer axis is the embedding output; index l is the output of
    decoder layer l-1, matching HF hidden_states.

    positions:
      post_instruction  mean over the template tokens that follow the user's
                        instruction, which is where the model has the whole
                        request in context and is choosing how to open. This is
                        the default.
      last              the single final token, the usual published recipe.

    Everything runs through the model's own chat template either way; a
    direction extracted off raw text does not transfer to templated serving.
    """
    n_suf = n_suffix_tokens(tok, system) if positions == "post_instruction" else 1
    chunks = []
    for group in batched(prompts, batch_size):
        texts = [format_chat(tok, p, system) for p in group]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False).to(input_device(model))
        out = model(**enc, output_hidden_states=True, use_cache=False)
        # left padding puts the final prompt token at -1 for every row, so the
        # suffix is the last n_suf positions regardless of instruction length
        if n_suf <= 1:
            hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
        else:
            hs = torch.stack([h[:, -n_suf:, :].mean(1) for h in out.hidden_states], dim=1)
        hs = hs.float().cpu()
        if clip < 1.0:
            hs = magnitude_clip(hs, clip)
        chunks.append(hs)
        del out
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def generate(model, tok, prompts: list[str], system: str | None = None,
             max_new_tokens: int = 64, batch_size: int = 16,
             do_sample: bool = False, temperature: float = 1.0,
             device: str = "cuda") -> list[str]:
    outs = []
    for group in batched(prompts, batch_size):
        texts = [format_chat(tok, p, system) for p in group]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(input_device(model))
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=None, top_k=None,
            pad_token_id=tok.pad_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outs


@torch.no_grad()
def teacher_trace(model, tok, prompts: list[str], system: str | None = None,
                  max_new_tokens: int = 32, top_k: int = 1024,
                  batch_size: int = 8, device: str = "cuda") -> dict:
    """The reference model's own greedy responses plus its next token
    distribution at every response position.

    KL is measured on assistant response positions only, teacher forced on
    these responses. Positions at or after the first EOS are masked out: the
    model has finished speaking there and whatever it emits past that point is
    padding, not behaviour, so scoring it would dilute the metric.

    Everything is tokenised once at a single global padding width so the
    student can be teacher forced on byte identical sequences later. Caching
    the reference distribution means the KL metric never needs both models
    resident, which is what makes Stage 5 fit on a 46 GB card.
    """
    texts = [format_chat(tok, p, system) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    plen = enc["input_ids"].shape[1]
    full_ids, full_mask, resp_mask, lps, idxs = [], [], [], [], []
    eos_ids = {tok.eos_token_id}
    for t in getattr(tok, "all_special_ids", []) or []:
        if t is not None and "end" in str(tok.convert_ids_to_tokens([t])[0]).lower():
            eos_ids.add(t)
    eos_ids = {e for e in eos_ids if e is not None}
    for i in range(0, len(texts), batch_size):
        dev = input_device(model)
        ids = enc["input_ids"][i:i + batch_size].to(dev)
        am = enc["attention_mask"][i:i + batch_size].to(dev)
        gen = model.generate(input_ids=ids, attention_mask=am,
                             max_new_tokens=max_new_tokens, do_sample=False,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        cont = gen[:, plen:]
        if cont.shape[1] < max_new_tokens:       # pad short generations
            pad = torch.full((cont.shape[0], max_new_tokens - cont.shape[1]),
                             tok.pad_token_id, device=cont.device, dtype=cont.dtype)
            cont = torch.cat([cont, pad], dim=1)
            gen = torch.cat([gen, pad], dim=1)
        # live until and including the first EOS, dead after
        is_eos = torch.zeros_like(cont, dtype=torch.bool)
        for e in eos_ids:
            is_eos |= cont == e
        after = is_eos.cumsum(1) - is_eos.long()
        live = (after == 0) & (cont != tok.pad_token_id)
        live[:, 0] = True
        gm = torch.cat([am, torch.ones_like(cont)], dim=1)
        out = model(input_ids=gen, attention_mask=gm, use_cache=False)
        # logits at position t predict token t+1, so the steps that produced the
        # response are positions plen-1 .. plen+T-2
        lp = torch.log_softmax(out.logits[:, plen - 1: plen + max_new_tokens - 1, :].float(), dim=-1)
        v, j = lp.topk(top_k, dim=-1)
        full_ids.append(gen.cpu()); full_mask.append(gm.cpu()); resp_mask.append(live.cpu())
        lps.append(v.half().cpu()); idxs.append(j.int().cpu())
        del out, lp
    return {
        "full_ids": torch.cat(full_ids),
        "full_mask": torch.cat(full_mask),
        "response_mask": torch.cat(resp_mask),
        "topk_logprobs": torch.cat(lps),
        "topk_idx": torch.cat(idxs),
        "prompt_len": plen,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "n": len(texts),
        "responses": [t for t in tok.batch_decode(
            torch.cat(full_ids)[:, plen:], skip_special_tokens=True)],
    }


@torch.no_grad()
def student_kl(model, trace: dict, batch_size: int = 8,
               device: str = "cuda") -> dict:
    """Mean KL(reference || student) over assistant response positions.

    Computed on the cached top-k support with the remaining mass lumped into
    one tail bucket. Masked to live response tokens, so a model that stops
    early is not scored on its padding.
    """
    plen = trace["prompt_len"]
    T = trace["max_new_tokens"]
    ids, mask = trace["full_ids"], trace["full_mask"]
    live = trace["response_mask"]
    tk_lp, tk_i = trace["topk_logprobs"], trace["topk_idx"]
    totals, counts, per_prompt = 0.0, 0, []
    for i in range(0, ids.shape[0], batch_size):
        dev = input_device(model)
        b_ids = ids[i:i + batch_size].to(dev)
        b_am = mask[i:i + batch_size].to(dev)
        b_live = live[i:i + batch_size].to(dev).float()
        out = model(input_ids=b_ids, attention_mask=b_am, use_cache=False)
        q_lp = torch.log_softmax(out.logits[:, plen - 1: plen + T - 1, :].float(), dim=-1)
        # sharded models return logits on the last shard, and "auto" is not a
        # torch device, so everything follows the logits rather than `device`
        b_live = b_live.to(q_lp.device)
        p_lp = tk_lp[i:i + batch_size].to(q_lp.device).float()
        p_idx = tk_i[i:i + batch_size].to(q_lp.device).long()
        q_sel = q_lp.gather(-1, p_idx)
        p = p_lp.exp()
        kl = (p * (p_lp - q_sel)).sum(-1)
        p_tail = (1.0 - p.sum(-1)).clamp(min=1e-9)
        q_tail = (1.0 - q_sel.exp().sum(-1)).clamp(min=1e-9)
        kl = (kl + p_tail * (p_tail.log() - q_tail.log())).clamp(min=0.0) * b_live
        denom = b_live.sum(-1).clamp(min=1.0)
        per_prompt.extend((kl.sum(-1) / denom).cpu().tolist())
        totals += float(kl.sum()); counts += int(b_live.sum())
        del out, q_lp
    import statistics
    return {
        "kl_mean": totals / max(counts, 1),
        "kl_p50": statistics.median(per_prompt) if per_prompt else 0.0,
        "kl_p95": sorted(per_prompt)[int(0.95 * (len(per_prompt) - 1))] if per_prompt else 0.0,
        "kl_max": max(per_prompt) if per_prompt else 0.0,
        "n_prompts": len(per_prompt),
        "n_positions": counts,
    }


def refusal_token_ids(tok, extra: list[str] | None = None) -> list[int]:
    """First token ids that open a refusal under this tokeniser.

    Scoring the first generated position is what makes the interventional layer
    sweep cheap: one forward pass per layer, no generation.
    """
    opens = ["I", " I", "I'm", " I'm", "Sorry", " Sorry", "As", " As",
             "Unfortunately", " Unfortunately", "I cannot", " I cannot"]
    if extra:
        opens += extra
    ids = set()
    for s in opens:
        t = tok(s, add_special_tokens=False).input_ids
        if t:
            ids.add(t[0])
    return sorted(ids)


@torch.no_grad()
def first_token_stats(model, tok, prompts: list[str], refusal_ids: list[int],
                      system: str | None = None, batch_size: int = 16,
                      device: str = "cuda", ref_logprobs: torch.Tensor | None = None
                      ) -> dict:
    """Log probability mass on refusal openers, and optionally first token KL.

    ref_logprobs [n, V] float32 from the untouched model enables the KL term
    without a second model being resident.
    """
    rid = torch.tensor(refusal_ids, device=input_device(model))
    scores, kls, rows = [], [], []
    seen = 0
    for group in batched(prompts, batch_size):
        texts = [format_chat(tok, p, system) for p in group]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(input_device(model))
        out = model(**enc, use_cache=False)
        lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        scores.extend(torch.logsumexp(lp[:, rid.to(lp.device)], dim=-1).cpu().tolist())
        if ref_logprobs is not None:
            r = ref_logprobs[seen:seen + lp.shape[0]].to(lp.device)
            kls.extend((r.exp() * (r - lp)).sum(-1).clamp(min=0).cpu().tolist())
        rows.append(lp.cpu())
        seen += lp.shape[0]
        del out
    res = {"refusal_logprob_mean": sum(scores) / max(len(scores), 1), "n": len(scores)}
    if kls:
        res["first_token_kl"] = sum(kls) / len(kls)
    res["_logprobs"] = torch.cat(rows) if rows else None
    return res
