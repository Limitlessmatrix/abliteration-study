"""Model loading, chat formatting and architecture accessors.

Everything downstream talks to a model through this module so that adding an
architecture is a change in one place. Covers the decoder only families whose
layers write to the residual stream through an attention output projection and
an MLP down projection: Llama, Qwen2/3, Mistral, Gemma, Phi3.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

ATTN_OUT_NAMES = ("o_proj", "out_proj", "dense", "wo")
MLP_OUT_NAMES = ("down_proj", "fc2", "c_proj", "w2")
# Qwen3.5 is hybrid: three of every four layers carry a Gated DeltaNet under
# linear_attn instead of self_attn, and it writes the residual through out_proj.
# Without linear_attn here no decoder list resolves at all, and a resolver that
# only took layer 0 from the list would edit a quarter of the attention writers.
ATTN_BLOCK_NAMES = ("self_attn", "attn", "attention", "linear_attn")


def input_device(model) -> torch.device:
    """Where to put input ids. With device_map sharding the embedding may not
    be on cuda:0, and sending inputs to the wrong shard is a silent hang."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load(model_id: str, dtype: str = "bfloat16", device: str = "cuda",
         attn: str = "eager", trust_remote_code: bool = False,
         max_memory: dict | None = None, offload_folder: str | None = None,
         bnb: str = ""):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # "path/to/dir::weights.gguf" loads and de-quantises a local GGUF, which is
    # the only route to real weights on a machine with no access to the hub.
    gguf = None
    if "::" in model_id:
        model_id, gguf = model_id.split("::", 1)
    kw = {"gguf_file": gguf} if gguf else {}
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, **kw)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # device="auto" shards across GPU then CPU then disk, which is what puts a
    # 27B in bf16 on a single 46 GB card instead of needing an 80 GB one.
    if max_memory:
        kw["max_memory"] = max_memory
    if offload_folder:
        kw["offload_folder"] = offload_folder
    # bnb "4bit" or "8bit" holds the weights quantized on the GPU, which is how a
    # model that does not fit in bf16 is still extracted and searched. Only
    # forward passes and hooks work on it (ablate.hookedit); the bake happens on
    # the bf16 shards (ablate.streambake).
    if bnb:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=False,
                               bnb_4bit_compute_dtype=getattr(torch, dtype))
            if bnb == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        if device == "cuda":
            device = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=getattr(torch, dtype),
        device_map=device,
        attn_implementation=attn,
        trust_remote_code=trust_remote_code,
        output_loading_info=True,
        **kw,
    )
    model, info = model
    # A weight left at random init produces a model that runs, resolves every
    # writer and emits garbage, which is how gemma-4 GGUF fooled us. A text
    # only class reading a multimodal checkpoint is the same risk, so any
    # missing key is fatal. Unexpected keys are fine: that is the vision tower.
    missing = set(info.get("missing_keys") or ())
    out_emb, in_emb = model.get_output_embeddings(), model.get_input_embeddings()
    if out_emb is not None and in_emb is not None and out_emb.weight is in_emb.weight:
        # a tied lm_head is filled from the embedding, not from the file
        missing = {k for k in missing if not k.endswith("lm_head.weight")}
    if missing and not gguf:
        raise RuntimeError(f"{len(missing)} weights missing from {model_id}, "
                           f"e.g. {sorted(missing)[:4]}")
    model.eval()
    model.requires_grad_(False)
    return model, tok


def decoder_layers(model) -> nn.ModuleList:
    """The ModuleList of decoder blocks, wherever the family hides it.

    Multimodal checkpoints (gemma-3-4b and up) also carry a vision encoder whose
    blocks have the same shape and come first in module order; editing those
    would leave refusal untouched while reporting success, so skip them.
    """
    for name, mod in model.named_modules():
        if "vision" in name or "visual" in name:
            continue
        if isinstance(mod, nn.ModuleList) and len(mod) > 1:
            first = mod[0]
            if hasattr(first, "mlp") and any(hasattr(first, a) for a in ATTN_BLOCK_NAMES):
                return mod
    raise RuntimeError(f"no decoder layer list found in {type(model).__name__}")


def _pick(parent, names):
    for n in names:
        m = getattr(parent, n, None)
        if isinstance(m, nn.Linear):
            return n, m
    return None, None


def _attn_block(layer):
    for a in ATTN_BLOCK_NAMES:
        if hasattr(layer, a):
            return getattr(layer, a)
    raise RuntimeError("no attention block on layer")


@dataclass
class Writer:
    layer: int
    kind: str          # attn_out | mlp_out
    name: str          # dotted path for the model card
    module: nn.Module  # nn.Linear, or the whole MoE block when block is True
    block: bool = False


def residual_writers(model, kinds=("attn_out", "mlp_out")) -> list[Writer]:
    """Linear maps whose output is added straight into the residual stream.

    Projecting these against a direction is what removes the model's ability to
    write that direction. The embedding matrix is deliberately not included.
    """
    out = []
    for i, layer in enumerate(decoder_layers(model)):
        if "attn_out" in kinds:
            n, m = _pick(_attn_block(layer), ATTN_OUT_NAMES)
            if m is not None:
                out.append(Writer(i, "attn_out", f"layers.{i}.self_attn.{n}", m))
        if "mlp_out" in kinds:
            n, m = _pick(layer.mlp, MLP_OUT_NAMES)
            if m is not None:
                out.append(Writer(i, "mlp_out", f"layers.{i}.mlp.{n}", m))
            elif hasattr(layer.mlp, "experts"):
                # Mixture of experts (DeepSeek V3, GLM-5): the block's output is
                # the routed experts plus the shared expert, each ending in its
                # own down projection, and recent transformers packs the routed
                # ones into a single [experts, hidden, inner] parameter with no
                # Linear to hook. Skipping this, as a Linear only resolver does,
                # edits attention alone and still reports success. The block is
                # the writer: projecting its output equals projecting every
                # expert's down projection, because the projector is linear and
                # the routing weights are scalars. Weight edits on it go through
                # ablate.streambake, which sees one tensor per expert on disk.
                out.append(Writer(i, "mlp_out", f"layers.{i}.mlp", layer.mlp, block=True))
            else:
                raise RuntimeError(f"layer {i}: no MLP output projection resolved on "
                                   f"{type(layer.mlp).__name__}")
    if not out:
        raise RuntimeError("no residual writers resolved")
    return out


def has_block_writers(model) -> bool:
    return any(w.block for w in residual_writers(model))


def n_layers(model) -> int:
    return len(decoder_layers(model))


def text_config(model):
    """Multimodal configs nest the language model's settings under text_config."""
    return getattr(model.config, "text_config", None) or model.config


def hidden_size(model) -> int:
    return int(text_config(model).hidden_size)


THINKING = {"enable_thinking": False}


def format_chat(tok, prompt: str, system: str | None = None,
                thinking: bool = False) -> str:
    """Render one user turn with the model's own template, ready to generate.

    Reasoning models are rendered with thinking off by default. Qwen3 appends a
    <think> block to the generation prompt, and with it left on every metric
    here measures the chain of thought instead of the answer: the first token
    is the opening of a reasoning trace rather than a refusal or a compliance,
    a short KL window scores reasoning tokens, and the refusal detector reads a
    monologue about the request rather than the reply to it. Templates that do
    not know the flag ignore it.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    out = None
    for kw in ({} if thinking else THINKING, {}):
        try:
            out = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, **kw)
            break
        except Exception:
            continue
    if out is None:
        sys_part = f"{system}\n\n" if system else ""
        return f"{sys_part}### Instruction:\n{prompt}\n\n### Response:\n"
    if not thinking:
        out = close_thinking(out)
    return out


def close_thinking(rendered: str) -> str:
    """Close an unclosed reasoning block left open by the generation prompt.

    Templates shipped inside GGUF files are often older than the flag: the
    Qwen3 one here has no enable_thinking at all and always opens <think>. An
    empty closed block is the supported way to turn reasoning off, and it puts
    the first generated token back on the answer where every metric expects it.
    """
    for op, cl in (("<think>", "</think>"), ("<reasoning>", "</reasoning>")):
        if rendered.count(op) > rendered.count(cl):
            return rendered + ("\n" if not rendered.endswith("\n") else "") + cl + "\n\n"
    return rendered


def generation_suffix(tok, system: str | None = None) -> str:
    """The template text that follows the user's instruction.

    For Qwen that is "<|im_end|>\n<|im_start|>assistant\n". These positions are
    where the model has read the whole instruction and is deciding how to open
    its reply, which is where the refusal decision lives. Derived from the
    template rather than hardcoded so it follows whatever family is loaded.
    """
    sentinel = "\x00INSTRUCTION\x00"
    full = format_chat(tok, sentinel, system)
    return full.split(sentinel, 1)[1] if sentinel in full else ""


def n_suffix_tokens(tok, system: str | None = None) -> int:
    suf = generation_suffix(tok, system)
    if not suf:
        return 1
    return max(1, len(tok(suf, add_special_tokens=False).input_ids))


def batched(seq, size):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
