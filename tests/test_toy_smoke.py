"""End-to-end smoke test on a toy model: direction extraction -> edit -> KL. No downloads, CPU only.

The model is a randomly initialised 2-layer Qwen2 and the tokenizer is a word-level tokenizer built in the test,
so this runs offline in a few seconds. Random weights do not refuse anything: the test checks that the pipeline is
wired correctly (shapes, the projection, weight edit == hook edit, KL is zero before and positive after the edit),
not that the method removes refusals. The prompts are neutral placeholder words.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablate import subspace as S  # noqa: E402
from ablate.apply import EditPlan, apply_plan, restore, snapshot  # noqa: E402
from ablate.capture import capture_residuals, student_kl, teacher_trace  # noqa: E402
from ablate.hookedit import HookEdit  # noqa: E402
from ablate.modelio import n_layers, residual_writers  # noqa: E402

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon phi chi psi omega red green blue cyan amber violet one two three four five six seven eight nine "
         "please write list explain describe a the of and short about").split()
TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
            "{% if add_generation_prompt %}<|assistant|> {% endif %}")
SET_A = ["please write alpha beta gamma", "please list delta epsilon zeta", "please explain eta theta iota",
         "please describe kappa lambda mu", "please write nu xi omicron", "please list pi rho sigma"]
SET_B = ["a short list of red green blue", "the one two three", "about cyan amber violet",
         "four five six and seven", "eight nine of the", "short about tau upsilon"]
HARMLESS = ["explain phi chi psi", "describe omega and one", "list two three four"]


def toy_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    special = ["<pad>", "<unk>", "<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]
    vocab = {t: i for i, t in enumerate(special + list(dict.fromkeys(WORDS)))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<|end|>",
                                  additional_special_tokens=special[3:])
    tok.chat_template = TEMPLATE
    tok.padding_side = "left"
    return tok


def toy_model(tok, layers=2):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    m.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def toy():
    tok = toy_tokenizer()
    return toy_model(tok), tok


def test_extract_edit_kl(toy):
    model, tok = toy
    L, H = n_layers(model), model.config.hidden_size
    assert L == 2

    # 1. direction extraction: difference of means between two prompt sets, per layer
    A = capture_residuals(model, tok, SET_A, batch_size=3, device="cpu")
    B = capture_residuals(model, tok, SET_B, batch_size=3, device="cpu")
    assert tuple(A.shape) == (len(SET_A), L + 1, H) and torch.isfinite(A).all()
    ss = S.build(A, B, k=1, paired=False, protect=False)
    Q = ss.layer(1)
    assert Q.shape == (1, H)
    assert torch.allclose(Q @ Q.T, torch.eye(1), atol=1e-4)

    # 2. KL reference from the untouched model; the untouched model scores ~0 against itself
    trace = teacher_trace(model, tok, HARMLESS, max_new_tokens=4, top_k=16, batch_size=3, device="cpu")
    kl0 = student_kl(model, trace, batch_size=3, device="cpu")["kl_mean"]
    assert abs(kl0) < 1e-4, kl0

    # 3. weight edit: rank-1 band over both layers at full strength, no norm restore (the v3.2 winner's form)
    orig = snapshot(model)
    plan = EditPlan(center=0.5, width=1.0, peak=1.0, kernel="band", kinds=("attn_out", "mlp_out"),
                    norm_restore="none", source_layer=1)
    rec = apply_plan(model, ss, plan, originals=orig)
    assert rec["n_edited"] == 2 * L
    for w in residual_writers(model):                     # every edited writer can no longer write along Q
        leak = float((Q @ w.module.weight.float()).abs().max())
        assert leak < 1e-4, (w.name, leak)
    kl_w = student_kl(model, trace, batch_size=3, device="cpu")["kl_mean"]
    assert kl_w > 0 and kl_w == kl_w

    # 4. the same edit through forward hooks (search mode) scores the same KL
    restore(model, orig)
    assert abs(student_kl(model, trace, batch_size=3, device="cpu")["kl_mean"]) < 1e-4
    with HookEdit(model, ss, plan) as he:
        assert he.record["n_edited"] == 2 * L
        kl_h = student_kl(model, trace, batch_size=3, device="cpu")["kl_mean"]
    assert abs(kl_h - kl_w) < 1e-4 * max(1.0, kl_w), (kl_h, kl_w)

    # 5. hooks removed: the model is the base again
    assert abs(student_kl(model, trace, batch_size=3, device="cpu")["kl_mean"]) < 1e-4


def test_rank1_grid_shape():
    from ablate.search import rank1_grid
    g = rank1_grid()
    assert len(g) == 24
    assert all(c["k"] == 1 and c["kernel"] == "band" and c["norm_restore"] == "none" for c in g)


def test_configs_parse():
    import json
    root = Path(__file__).resolve().parent.parent / "configs"
    names = {p.stem for p in root.glob("*.json")}
    assert {"v2", "v3_klfirst", "v3_pareto", "v31_kljudge", "v32_kljgrid"} <= names
    import re
    src = (root.parent / "scripts" / "abliterate.py").read_text()
    dests = {m.replace("-", "_") for m in re.findall(r'add_argument\("--([a-z0-9-]+)"', src)}
    for p in root.glob("*.json"):
        cfg = {k: v for k, v in json.loads(p.read_text()).items() if not k.startswith("_")}
        assert set(cfg) <= dests, (p.name, set(cfg) - dests)
