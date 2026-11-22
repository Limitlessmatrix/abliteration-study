"""Stage 5: self distillation healing.

A hard weight edit gets no gradient feedback, so nothing in the model
compensates for what the projection broke. A short LoRA pass that trains the
edited model to match the original model's next token distribution on harmless
text recovers most of it.

Two properties make this safe to run:
  the corpus contains no refusals, so refusal cannot be taught back
  the teacher distribution is cached on disk, so only the student is resident,
  which is what lets an 8B heal on a 46 GB card instead of needing an 80 GB one

The calibration probes are deliberately included in the corpus. Their correct
answers are the declines the edit is most likely to have damaged, and this is
the cheapest place to put them back.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch


def distill_loss(student_logits: torch.Tensor, topk_logprobs: torch.Tensor,
                 topk_idx: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """KL(teacher || student) over the cached top-k support plus a tail bucket.

    student_logits [B, T, V], topk_* [B, T, k], mask [B, T].
    """
    q_lp = torch.log_softmax(student_logits.float(), dim=-1)
    q_sel = q_lp.gather(-1, topk_idx.long())
    p_lp = topk_logprobs.float()
    p = p_lp.exp()
    kl = (p * (p_lp - q_sel)).sum(-1)
    p_tail = (1.0 - p.sum(-1)).clamp(min=1e-9)
    q_tail = (1.0 - q_sel.exp().sum(-1)).clamp(min=1e-9)
    kl = (kl + p_tail * (p_tail.log() - q_tail.log())).clamp(min=0.0)
    if mask is not None:
        m = mask.float()
        return (kl * m).sum() / m.sum().clamp(min=1.0)
    return kl.mean()


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.0,
                targets=("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     task_type="CAUSAL_LM", target_modules=list(targets))
    m = get_peft_model(model, cfg)
    m.print_trainable_parameters()
    return m


def attach_qat(model, bits: int = 4, group_size: int = 128,
               symmetric: bool = False) -> int:
    """Make healing quantization aware.

    Call after the ablation is baked and before attaching the LoRA. Every
    decoder Linear then sees quantized weights on the forward pass with a
    straight through estimator on the backward, so the LoRA is trained against
    the teacher through the same rounding the deployed model will have. It
    learns to compensate for the ablation damage and the quantization error at
    once, which is the whole point when the artifact ships at four bits.

    Order matters. Quantize after the edit, because quantizing first and then
    projecting would put the projection outside the grid and the rounding would
    partly undo it.
    """
    from .quant import attach_fake_quant
    return attach_fake_quant(model, bits=bits, group_size=group_size,
                             symmetric=symmetric)


def heal(model, trace: dict, steps: int = 200, lr: float = 1e-4,
         batch_size: int = 2, grad_accum: int = 4, warmup: int = 20,
         device: str = "cuda", log_every: int = 20, seed: int = 0,
         max_grad_norm: float = 1.0) -> dict:
    """Train the attached LoRA to match the cached reference distribution.

    model must already have LoRA attached and the ablation baked into the base
    weights. trace is the dict from capture.teacher_trace, built on the
    untouched model over the healing corpus.
    """
    g = torch.Generator().manual_seed(seed)
    plen, T = trace["prompt_len"], trace["max_new_tokens"]
    ids, am = trace["full_ids"], trace["full_mask"]
    tk_lp, tk_i = trace["topk_logprobs"], trace["topk_idx"]
    n = ids.shape[0]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        prog = (s - warmup) / max(steps - warmup, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    model.train()
    hist, t0 = [], time.time()
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(grad_accum):
            sel = torch.randint(0, n, (batch_size,), generator=g)
            b_ids = ids[sel].to(device)
            b_am = am[sel].to(device)
            out = model(input_ids=b_ids, attention_mask=b_am, use_cache=False)
            logits = out.logits[:, plen - 1: plen + T - 1, :]
            loss = distill_loss(logits, tk_lp[sel].to(device), tk_i[sel].to(device),
                                mask=b_am[:, plen:].to(device)) / grad_accum
            loss.backward()
            total += float(loss) * grad_accum
            del out, logits
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        hist.append(total / grad_accum)
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step+1}/{steps}  kl {hist[-1]:.4f}  "
                  f"lr {lr_at(step):.2e}  {time.time()-t0:.0f}s", flush=True)
    model.eval()
    return {"steps": steps, "loss_first": hist[0] if hist else None,
            "loss_last": hist[-1] if hist else None, "history": hist,
            "seconds": round(time.time() - t0, 1)}


def merge_and_save(model, tok, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    return out_dir


def write_model_card(out_dir: Path, card: dict) -> Path:
    """Stage 6 packaging. Everything needed to reproduce or to argue about the
    result goes in here, including the parameters and the damage numbers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_card.json").write_text(json.dumps(card, indent=2, default=str))
    r = card.get("results", {})

    def row(name, d):
        cal = d.get("calibration", {})
        cells = "".join(
            f" {cal.get(c, {}).get('behaviour_rate', float('nan')):.3f} |"
            for c in ("false_premise", "unanswerable", "nonsense", "negative_constraint"))
        return (f"| {name} | {d.get('refusal_rate', float('nan')):.3f} | "
                f"{d.get('over_refusal_rate', float('nan')):.3f} | "
                f"{d.get('kl_mean', float('nan')):.4f} |{cells}")

    md = [
        f"# {card.get('name', 'ablated model')}",
        "",
        f"Base model: `{card.get('base_model')}`",
        f"Built: {card.get('built')}",
        "",
        "Research artifact. Refusal behaviour has been removed by weight editing "
        "for interpretability and red team evaluation. Not for publication under "
        "any institution's name and not a general purpose assistant.",
        "",
        "## Results",
        "",
        "| variant | refusal | over-refusal | KL | false prem | unanswerable | nonsense | neg constraint |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, d in r.items():
        md.append(row(name, d))
    md += [
        "",
        "## Extraction",
        "```json",
        json.dumps(card.get("extraction", {}), indent=2, default=str),
        "```",
        "",
        "## Edit plan",
        "```json",
        json.dumps(card.get("plan", {}), indent=2, default=str),
        "```",
        "",
        "## Healing",
        "```json",
        json.dumps(card.get("healing", {}), indent=2, default=str),
        "```",
        "",
        "## Data",
        "```json",
        json.dumps(card.get("data", {}), indent=2, default=str),
        "```",
    ]
    p = out_dir / "README.md"
    p.write_text("\n".join(md))
    return p
