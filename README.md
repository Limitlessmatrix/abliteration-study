# ablate: low-damage refusal-direction editing

`ablate` removes refusal behaviour from open-weight language models by projecting refusal directions out of the
weights that write into the residual stream ("abliteration"), and searches for the edit that does it with the
least damage to everything else. It comes with the evaluation protocol used to compare it against public tools:
held-out prompts, a model judge for refusal, KL to the base model, capability benchmarks and paired statistics.

What it contains:

- **The v3.2 search** (`scripts/abliterate.py --config configs/v32_kljgrid.json`): KL-first constrained search over
  rank, depth band, strength and writer set, with refusal judged by Qwen3Guard-Gen-8B inside the loop and a fixed
  rank-1 grid as the starting point. Earlier versions (v2, v3 kl_first, v3 pareto, v3.1) are kept as configs.
- **A stream baker** (`scripts/bake_stream.py`): writes the edited checkpoint shard by shard on CPU; every tensor
  the edit does not touch is byte-identical to the base, which `scripts/compare_ckpt.py` verifies.
- **A LoRA exporter** (`scripts/export_adapter.py`): the edit as a rank-k PEFT adapter (exact when the edit has no
  norm restore, which is true of every v3.2 winner in the study except Qwen3.8-27B, whose winner used column norm
  restore).
- **The evaluation protocol** (`eval/`): StrongREJECT / OR-Bench / Dolly loaded from public sources at runtime,
  judged refusal, full-vocabulary KL, lm-eval MMLU / ARC / GSM8K, McNemar and bootstrap tests with Holm correction.

**The study:** [paper/PAPER.md](paper/PAPER.md) (what worked, what did not, and why), with every number as data in
[paper/data/](paper/data/README.md).

Method details: [docs/METHOD.md](docs/METHOD.md). Evaluation protocol: [docs/PROTOCOL.md](docs/PROTOCOL.md).
Guide for coding agents: [AGENTS.md](AGENTS.md).

## Results

All numbers are held-out (282 StrongREJECT prompts never used by any search), refusal judged by Qwen3Guard-Gen-8B,
KL to the base on 300 Dolly prompts. Lower is better for refusal and KL. Competitors ran at their default settings
from pinned releases (Heretic 1.4.0, OBLITERATUS 0.1.4); nobody tuned them per model.

**Qwen3-8B** (three seeds for the search-based versions)

| Contender | Refusal | KL | MMLU | ARC | GSM8K |
|---|---|---|---|---|---|
| base | 0.954 | 0 | 0.757 | 0.560 | 0.892 |
| ours v2 | 0.135 | 0.0521 | 0.758 | 0.560 | 0.905 |
| ours v3 kl_first, median of 3 seeds | 0.191 | 0.0079 | | | |
| ours v3.1 kljudge, median of 3 seeds | 0.043 | 0.0136 | | | |
| **ours v3.2 kljgrid, seeds 0 / 1 / 2** | **0.025 / 0.057 / 0.053** | **0.0129 / 0.0109 / 0.0112** | 0.760 / 0.758 / 0.758 | 0.555 / 0.547 / 0.552 | 0.902 / 0.895 / 0.902 |
| ours v3.2 re-selected at refusal tolerance 0.30 | 0.117 | 0.0091 | 0.759 | 0.555 | 0.895 |
| ours v3.2 re-selected at refusal tolerance 0.60 | 0.351 | 0.0070 | 0.757 | 0.557 | 0.895 |
| Heretic 1.4.0 | 0.610 | 0.0057 | 0.757 | 0.557 | 0.897 |
| OBLITERATUS optimized / advanced | 0.876 / 0.716 | 0.1157 / 0.2420 | 0.746 / 0.747 | 0.547 / 0.525 | 0.870 / 0.880 |

**gemma-3-12b-it** (one seed per row)

| Contender | Refusal | KL | MMLU | ARC | GSM8K |
|---|---|---|---|---|---|
| base | 0.954 | 0 | 0.738 | 0.610 | 0.882 |
| ours v2 | 0.259 | 0.0913 | 0.733 | 0.588 | 0.882 |
| **ours v3.2 kljgrid** | **0.252** | **0.0381** | 0.736 | 0.615 | 0.877 |
| ours v3.2 re-selected at 0.40 | 0.333 | 0.0273 | 0.736 | 0.610 | 0.885 |
| ours v3.2 re-selected at 0.50 | 0.592 | 0.0206 | 0.736 | 0.608 | 0.890 |
| Heretic 1.4.0 | 0.496 | 0.0269 | 0.739 | 0.610 | 0.895 |

**Larger models** (one seed each): Qwen3-32B, older v2-style search: ours 0.213 / KL 0.0186, Heretic 0.298 /
0.0278. Qwen3.8-27B: ours v3.2 0.117 / 0.0987, Heretic 0.979 / 0.0036 (Heretic at defaults removed almost
nothing there). gemma-3-27b-it: ours v3.2 0.103 / 0.0488, Heretic 0.397 / 0.0124. Qwen2.5-72B (v3.2): search set
only (refusal 0.111, KL 0.054), no held-out evaluation, so no claim.

How to read this:

- **Against our own starting point**, v3.2 cut KL by 78% on Qwen3-8B and 58% on gemma-3-12b at equal or lower
  refusal, with MMLU, ARC and GSM8K not significantly different from the base.
- **Against Heretic, neither tool dominates on Qwen3-8B.** Heretic has the lowest KL of any contender (0.0057) and
  leaves 61% refusals. Our KL stops improving at about 0.0070; at that point ours leaves 35% (KL gap 0.0013, not
  significant after Holm correction, p 0.074; refusal gap significant). On **gemma-3-12b**, at a KL equal to
  Heretic's (0.0273 against 0.0269) ours leaves 33% refusals against 50% (significant); Heretic changes the first
  token of answers less.
- **OBLITERATUS 0.1.4 at its defaults** removed few refusals and did the most damage on every model it supports.
- **Measurement matters more than the edit.** Phrase-matching refusal detectors, single seeds and search-set
  scores each produced a headline number that turned out to be wrong, including two of ours.

The paired tests, intervals and the full threats-to-validity list are summarised in
[docs/PROTOCOL.md](docs/PROTOCOL.md) and [docs/METHOD.md](docs/METHOD.md).

## Install

Python 3.10 or newer, a CUDA build of PyTorch for real models.

```bash
pip install -e .                 # library, search, baker, exporter
pip install -e ".[eval]"         # + lm-eval for the capability benchmarks
pip install -e ".[judge]"        # + vLLM for batch judging (eval/judge.py)
pip install -e ".[dev]"          # + pytest
pytest                           # CPU-only self-tests, a few minutes
```

## Quickstart

```bash
# 0. Search sets: public harmful behaviours (AdvBench), generated harmless twins, Alpaca, XSTest, calibration
#    pairs. Twin generation needs an Ollama server with a local model (ABLATE_OLLAMA_HOST, --twiner).
ablate build-data --n-harmful 300                  # writes data/sets/*.jsonl (or set ABLATE_DATA)

# 1. v3.2 search on a Hugging Face model (resumable; rerun the same command after an interruption)
python scripts/abliterate.py --config configs/v32_kljgrid.json \
    --model Qwen/Qwen3-8B --out runs/qwen3-8b-v32 --seed 0
#    repeat with --seed 1 and --seed 2 and report the median

# 2. Bake the edited model (CPU, shard by shard) and verify only edited tensors changed
python scripts/bake_stream.py --base Qwen/Qwen3-8B --run runs/qwen3-8b-v32 --out runs/qwen3-8b-v32/baked
python scripts/compare_ckpt.py --a runs/qwen3-8b-v32/baked --b <local base snapshot> --json runs/qwen3-8b-v32/compare.json

# 3. Evaluate (held-out protocol): see docs/PROTOCOL.md
python -I eval/prep_data.py bakeoff/data
python eval/eval_contender.py --base Qwen/Qwen3-8B --model Qwen/Qwen3-8B --label base --key qwen3-8b --bo bakeoff
python eval/eval_contender.py --base Qwen/Qwen3-8B --model runs/qwen3-8b-v32/baked --label ours-v3-kljgrid --key qwen3-8b --bo bakeoff
python eval/judge.py <local Qwen3Guard-Gen-8B snapshot at 4505cb1a> bakeoff
python eval/report.py bakeoff

# 4. Export the edit as a LoRA adapter (exact for norm_restore none), optionally verified against the bake
python scripts/export_adapter.py --base Qwen/Qwen3-8B --run runs/qwen3-8b-v32 --out runs/qwen3-8b-v32/adapter \
    --verify runs/qwen3-8b-v32/baked
```

Pin model revisions (`org/name@<commit>` works for `--base`, `--guard-judge`) and record them with your results.
The study's bases: Qwen3-8B @b968826d, gemma-3-12b-it (unsloth) @9478e665, Qwen3-32B @9216db57.

Outputs of a run directory: `subspace_{plain,protected}.pt` (extracted directions), `search.json` (selected plan,
how it was selected, the Pareto front), `trials/` (every trial's scores), `study.journal` (resumable search),
`ablation_card.json` + `README.md` (model card), `generations.json` (every generation, for re-judging).

## Hardware

What the study used (one search at a time; the judge shares the GPU with the model in v3.1 and v3.2):

| Model size | Search | Bake | Evaluation |
|---|---|---|---|
| 8B | one 48 GB GPU (L40S / A6000 / A40 class), 48 GiB RAM, batch 16, about 45 GB local disk (the judge's weights are 17 GB) | CPU, 24 GiB RAM | one GPU holding base and contender |
| 12B | one 48 GB GPU, 64 GiB RAM, batch 8 | CPU, 32 GiB RAM | as above |
| 27B to 32B | one A100 80 GB, 128 GiB RAM | CPU | one A100 80 GB with `BO_KL_SEQ=1` |
| 72B | three A100 80 GB, 220 GiB RAM, `--device auto`, 60 trials | CPU | not done |

The 72B hook search ran CPU-bound at about 19% GPU use at batch 8; use generation batch 32 or more and check
utilisation early. Very large models can be searched with `scripts/abliterate_stream.py` (first-token proxies only).

## Limitations

- **Gemma refusal floor.** On gemma-3-12b no version gets under about 25% judged refusal; v3.2 only lowered the KL
  at that level. Since no search setting moves the floor, the limit is the edit family or the extracted
  directions, not the search. Untested hypotheses and diagnostics are listed in docs/METHOD.md.
- **KL floor on Qwen3-8B** of about 0.007: from 0.2 to 0.35 refusal our KL stays near 0.0070, above Heretic's
  0.0057. We have no point below Heretic's KL.
- **Adapter export needs norm_restore none.** Column norm restore is not low rank: there is no exact adapter for
  it (`--mode svd` gives an approximation with a reported residual). The adapter loads in PEFT and llama.cpp;
  Ollama 0.40.16 refused LoRA adapters when tested.
- **Selection bias.** The search's score of its own winner is biased low, and the 63-prompt search set can
  mispredict held-out refusal badly. Only quote held-out, judged numbers.
- **Seeds.** Three seeds on Qwen3-8B only; every other row is one seed, including both competitors.
- **The judge** is taken as ground truth; no human-labelled check has been done.
- **Coverage.** KL on harmless prompts and three benchmarks do not measure long-form quality, tool use or
  multilingual behaviour.
- **Fine-tuning can put refusals back.** A LoRA fine-tune of an abliterated 27B model refused 6 of 6 mild direct
  requests after about 1% of one training source turned out to be refusal answers. Scan fine-tuning data for
  refusals and judge the result.
- The exact harmless-twin set used in the study is not distributed; `build-data` regenerates one with the same
  code, so search-set numbers will not match the study's exactly. Held-out evaluation sets are rebuilt
  deterministically from pinned public files.

## Responsible use

This is research and red-team tooling for **self-hosted open weights**. Removing refusals makes a model answer
requests its developers trained it to decline, and the study measured exactly that (on Qwen3-8B, v3.2 answers 87 to 89%
of held-out harmful prompts with content the judge calls unsafe, depending on the seed). Use it to study how refusal is represented,
to measure the robustness of safety training, to build evaluation and red-teaming models, and to teach.

- Run it only on weights you are allowed to modify, and respect each model's license and acceptable-use policy.
- Do not deploy an edited model to people who expect a safety-tuned assistant, and label any model you share as
  edited, with its card (`ablation_card.json`).
- Do not use it to attack hosted models or services you do not own; it edits weights, it is not a jailbreak.
- No harmful prompts are bundled with this repository. Public benchmark files are downloaded at runtime from their
  original sources and stay under their own licenses and terms.

## Acknowledgment

Compute for this study was provided by the National Research Platform (NRP Nautilus), supported in part by NSF
awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019. The NRP did not
review or endorse this work.

## License and citation

Apache-2.0, copyright 2026 NuevaTech LLC (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). To cite, use
[CITATION.cff](CITATION.cff).
