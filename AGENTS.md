# AGENTS.md: guide for coding agents

Repository: `ablate`, low-damage refusal-direction editing plus its evaluation protocol. Read README.md for
results and limits, docs/METHOD.md for how the edit and search work, docs/PROTOCOL.md for evaluation rules.

## Layout

| Path | What |
|---|---|
| `ablate/` | library (importable package) |
| `ablate/config.py` | env vars, pinned judge revision, Hub id resolution |
| `ablate/datasets.py`, `sources.py`, `twins.py`, `calibration.py` | search-set building (public downloads + generated twins) |
| `ablate/capture.py`, `subspace.py` | residual capture, difference subspaces, protected variant |
| `ablate/apply.py`, `hookedit.py` | weight edit (`apply_plan`) and the identical hook edit (`HookEdit`) |
| `ablate/evalharness.py`, `refusal.py`, `guardjudge.py` | in-search scoring: refusal, KL, calibration, coherence |
| `ablate/search.py` | Optuna search, objectives (`scalar`, `pareto`, `kl_first`), `rank1_grid`, `select_trial` |
| `ablate/streambake.py`, `adapter.py` | CPU shard-by-shard bake; LoRA export |
| `ablate/streamsearch.py` | first-token proxy search for models that do not fit (proxy only) |
| `ablate/heal.py`, `quant.py`, `layerscan.py`, `pipeline.py`, `cli.py` | optional healing, quant presets, layer scan, older CLI |
| `scripts/abliterate.py` | main pipeline: extract -> baseline -> layer scan -> search -> final scoring -> card |
| `scripts/bake_stream.py`, `compare_ckpt.py`, `export_adapter.py`, `abliterate_stream.py` | bake, verify, export, big-model search |
| `scripts/selftest*.py` | self-tests on tiny random models (run by pytest) |
| `eval/` | held-out protocol: `prep_data.py`, `eval_contender.py`, `judge.py`, `report.py`, `repalign.py`, `checks.py`, `gguf_arms.py` |
| `configs/*.json` | argument defaults of each study version |
| `tests/` | pytest: toy end-to-end smoke test, wrappers for every self-test |

## Commands

```bash
pip install -e ".[dev]"; pytest                                  # CPU, no model downloads except one tokenizer (skipped if offline)
ablate build-data --n-harmful 300                                # search sets -> $ABLATE_DATA or ./data (needs Ollama for twins)
python scripts/abliterate.py --config configs/v32_kljgrid.json --model ORG/MODEL --out runs/NAME [--seed N]
python scripts/bake_stream.py --base ORG/MODEL[@REV] --run runs/NAME --out runs/NAME/baked
python scripts/compare_ckpt.py --a runs/NAME/baked --b BASE_DIR --json runs/NAME/compare.json
python scripts/export_adapter.py --base ORG/MODEL[@REV] --run runs/NAME --out runs/NAME/adapter [--verify runs/NAME/baked]
python -I eval/prep_data.py WORK/data [OBLITERATUS_TREE]
python eval/eval_contender.py --base BASE --model MODEL --label LABEL --key KEY --bo WORK   # label "base" first
python eval/judge.py JUDGE_DIR WORK; python eval/report.py WORK
```

## Config keys (scripts/abliterate.py; JSON keys = argument names with underscores)

| Key | Meaning | v3.2 value |
|---|---|---|
| `pairs`, `n_eval`, `cal` | extraction pairs, search-set size cap, calibration items | 200, 80, 96 |
| `k`, `contrast`, `clip` | directions extracted per layer, contrast, activation winsorising | 8, bulk, 1.0 |
| `edit_mode` | `weights` or `hook` (no weight copies; bake afterwards) | hook |
| `objective` | `scalar` (v2), `pareto`, `kl_first` | kl_first |
| `trials`, `seed`, `grid` | Optuna trials, TPE seed, `rank1` grid start | 150, 0, rank1 |
| `refusal_tol`, `cal_tol`, `kl_target` | selection tolerances; KL target (pareto) | 0.05, 0.05, 0.01 |
| `guard_judge` | Qwen3Guard dir or `repo@rev`; empty = phrase matcher | Qwen/Qwen3Guard-Gen-8B@4505cb1a... |
| `study_journal` | Optuna journal; `auto` = `<out>/study.journal` | auto |
| `quant`, `final_quant` | quantized scoring presets; must be empty in hook mode | "", "" |
| `batch`, `guard_batch` | generation batch, judge batch | 16 (8 for 12B), 16 |

Env: `ABLATE_DATA`, `ABLATE_OLLAMA_HOST`, `ABLATE_K_MAX` (rank ceiling, default 4), `ABLATE_JOURNAL`,
`BO_DIR`, `BO_KL_SEQ`, `BO_KL_CACHE`, `BO_DEVICE_MAP`, `BO_GEN_BATCH`, `LLAMA_SERVER`.

## Invariants (do not break)

1. Never compare tools on search-set refusal. Use held-out data, the Qwen3Guard judge, and paired tests with Holm
   correction (`eval/report.py`). Search-set refusal is biased low and mispredicts held-out refusal.
2. Never use phrase-matched refusal as a result. It is kept beside the judged rate only as a second opinion.
3. Report medians over at least three seeds for any search-based configuration; one seed is a draw, not a result.
4. Compare at matched refusal. Claim less damage only where KL is lower at held-out refusal equal or lower.
5. Same base tokenizer, chat template, greedy decoding and GSM8K batch for every contender of one base. Run the
   `base` label first; its answers are the KL reference.
6. A bake must leave every non-edited tensor byte-identical to the base; verify with `compare_ckpt.py` (number of
   differing tensors == number edited).
7. `HookEdit` and `apply_plan` must stay numerically equivalent (`selftest_big.py` checks it). Any change to the
   projection goes into both, plus `streambake.py` and `adapter.py`.
8. Never run two processes on one run directory or journal. A rerun of the same command resumes.
9. Do not change the protocol constants (datasets, judge revision, token limits, lm-eval version, limits) without
   saying so in the results; numbers produced under different settings are not comparable.
10. Do not add harmful prompt text to the repository. Load public datasets at runtime by their source URLs or ids.
11. Changing `ablate/search.py`, `evalharness.py`, `apply.py`, `subspace.py` or `capture.py` changes the method:
    run `pytest`, and treat results as a new version.

## Conventions

- Stdlib + torch + transformers + optuna; no new heavy dependencies in `ablate/` without need.
- Every stage writes its outputs atomically and skips work whose output exists.
- Self-tests are plain scripts that exit non-zero on failure; add a pytest wrapper in `tests/` for new ones.
