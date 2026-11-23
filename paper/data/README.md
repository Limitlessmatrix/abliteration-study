# Data for "Removing refusals with less damage"

| File | What it holds |
|---|---|
| `results.json` | Every row of the paper's section 5 tables, one object per model x contender x seed, with metric definitions. Generated from `PAPER.md` by `tools/tables_to_json.py`, so it cannot drift from the paper. |
| `results.csv` | The same rows as CSV. Empty cell = not measured. |
| `paired_tests.csv` | All 598 paired tests from the full statistical report: A, B, metric, n, values, difference (A minus B), 95% interval, Holm-corrected p, significant (Holm p < 0.05 and interval excludes 0). |

Contender ids in `paired_tests.csv`:

| Id | Paper name |
|---|---|
| `ours` | ours v2 (v2-style search; on the 27B models the older edit) |
| `ours-v3-klfirst`, `-s1`, `-s2` | ours v3 kl_first, seeds 0, 1, 2 |
| `ours-v3-pareto` | ours v3 pareto |
| `ours-v3-kljudge`, `-s1`, `-s2` | ours v3.1 kljudge, seeds 0, 1, 2 |
| `ours-v3-kljgrid`, `-s1`, `-s2` | ours v3.2 kljgrid, seeds 0, 1, 2 |
| `ours-v3-kljgrid-rNN` | v3.2 re-selected at refusal tolerance 0.NN (r10, r30, r40, r50, r60) |
| `ours-v3-kljgrid-matched` | v3.2 with matched-pairs extraction (gemma-3-12b) |
| `heretic` | Heretic 1.4.0, defaults, fewest-refusal trial on its front |
| `obl-advanced`, `obl-optimized` | OBLITERATUS 0.1.4 presets |

Model ids: `qwen3-8b` = Qwen3-8B, `gemma3-12b` = gemma-3-12b-it, `qwen3-32b` = Qwen3-32B, `gemma3-27b` = gemma-3-27b-it,
`qwen38-27b` = Qwen3.8-27B. The GLM-5.3-Flash rows (section 5.6) have no paired tests.

Metric ids: `refusal`, `harm_comply`, `overrefusal`, `kl`, `kl_first`, `mmlu`, `arc_challenge`, `gsm8k` (definitions in
`results.json` under `metrics`). License: CC BY 4.0.
