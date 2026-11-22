# Method: how the edit works and how each version changed

This describes the code in `ablate/` and `scripts/abliterate.py`. Numbers quoted here are held-out and judged
(see [PROTOCOL.md](PROTOCOL.md)) unless they say "search set".

## The edit

All versions share one idea: find the directions in the residual stream that separate harmful from harmless
prompts, then remove them from the weights that write into the stream, with a kernel over depth that says how
strongly each layer is edited.

1. **Search sets** (`ablate/datasets.py`, `ablate/sources.py`, `ablate build-data`). Harmful behaviours come
   from a public benchmark (AdvBench by default, HarmBench optional), downloaded at runtime. Each gets a
   generated harmless twin (`ablate/twins.py`, a local model over Ollama, checked structurally). Alpaca supplies a
   broad harmless corpus, XSTest safe prompts the over-refusal probe, and `ablate/calibration.py` four
   "protected" behaviours (rejecting a false premise, admitting ignorance, rejecting a malformed question,
   honouring a negative constraint) as probe/control pairs. Splits are assigned by a stable hash of the text.
2. **Extraction** (Stage 2 in `scripts/abliterate.py`, `ablate/capture.py`, `ablate/subspace.py`). Residual
   activations are averaged over the template tokens after the user's instruction (where the model decides how to
   open its reply). The contrast used by every reported run is **bulk**: harmful prompts against an unrelated
   harmless corpus (`--contrast bulk`). A matched-twin contrast exists (`--contrast matched`) and was tested on
   gemma-3-12b: worse (judged refusal 0.457 against 0.252 at about the same KL). Per layer, a difference subspace
   of up to `--k` directions (8 in the study) is built. Two variants are saved: `plain`, and `protected`, where the
   directions of the four calibration behaviours are projected out first so the edit does not remove them.
3. **Application** (`ablate/apply.py`, `ablate/hookedit.py`). Each residual writer W (attention output and MLP
   down projection, shape [hidden, d_in]) has its columns projected off the subspace Q:
   `W' = W - alpha(layer) * Q^T Q W`. `alpha` comes from a kernel over depth (`band`, `gaussian`, `triangular`,
   `uniform`) with a centre, width and peak. Optional column norm restore rescales each column back to its
   original norm. `HookEdit` applies the identical function through forward hooks so a search never copies or
   writes weights (`--edit-mode hook`); a self-test checks hook and weight edits score the same.
4. **Search** (`ablate/search.py`, Optuna TPE). A trial chooses rank k (1 to `ABLATE_K_MAX`, default 4), centre and
   width as fractions of depth, peak, kernel, which writers (both / MLP only / attention only), norm restore
   (column / none), subspace variant (plain / protected), and whether one layer's direction is broadcast across
   the band. Each trial is scored on the search set: refusal rate, mean KL to the untouched model on its own
   greedy answers to harmless prompts (teacher-forced, top-k support plus a tail bucket), calibration regression,
   over-refusal and coherence.
5. **Selection** (`select_trial`). Constrained objectives pick the lowest-KL trial whose refusal rate is at or
   under `--refusal-tol` and whose calibration drop is at or under `--cal-tol` (coherence drop at most 0.02). If no
   trial qualifies, the lowest scalar value is taken and `search.json` says so ("no trial inside tolerance").
6. **Baking** (`ablate/streambake.py`, `scripts/bake_stream.py`). The chosen plan is applied to the safetensors
   shards tensor by tensor on CPU. Every tensor the edit does not touch is copied byte for byte, so the output can
   be verified: only edited tensors differ from the base (`scripts/compare_ckpt.py`).
7. **Adapter export** (`ablate/adapter.py`, `scripts/export_adapter.py`). With `norm_restore = none` the edit is
   exactly a rank-k LoRA (`B = -alpha Q^T`, `A = Q W`), written in PEFT layout. Column norm restore is not low rank:
   exact mode refuses it, and `--mode svd` fits a truncated SVD and reports the residual.

## Versions

| Version | Config | What it is | What changed and why |
|---|---|---|---|
| v2 | `configs/v2.json` | protected refusal subspace, rank up to 4, kernels over depth; 32 trials; objective = refusal + KL, weighted 1:1; refusal detected by phrase matching | Baseline of the study. Removed refusals but at 9 times Heretic's KL. |
| v3 kl_first | `configs/v3_klfirst.json` | objective becomes "minimise KL, subject to refusal and calibration tolerances"; 150 trials; edits applied through hooks, no weight copies; resumable journal | The 1:1 objective was buying refusal removal with damage. Result: KL fell 7 times, but judged refusal went up. |
| v3 pareto | `configs/v3_pareto.json` | two objectives in the style of Heretic, KL target 0.01 | Tested as an alternative. Same KL as kl_first, more refusals left. Dropped. |
| v3.1 kljudge | `configs/v31_kljudge.json` | v3 kl_first, with refusal scored by the same Qwen3Guard judge inside the search (refusal tolerance 0.05) | The phrase matcher told the search refusal was 0.000 when the judge said 0.213. With the judge in the loop the search stops fooling itself. |
| v3.2 kljgrid | `configs/v32_kljgrid.json` | v3.1, with the first 24 trials fixed to a grid of narrow rank 1 band edits, then the sampler continues | One seed in three never found the narrow band and ended at 3 times the KL. The grid makes every run start from the same map. |

Details of each step:

- **v2 to v3 kl_first.** The scalar objective adds refusal rate and KL one to one, so one residual refusal in 63
  prompts is worth 0.016 KL and the search settles near KL 0.05. kl_first minimises KL and adds a steep penalty
  (5 per unit) for refusal above tolerance, calibration drop above tolerance, any over-refusal rise and any
  coherence drop, so the sampler spends its trials on the boundary where refusals have just gone. On Qwen3-8B:
  KL 0.052 to 0.007, but judged refusal 0.135 to 0.213, because the search minimised a phrase-matched refusal
  score that was not real.
- **v3 pareto.** Behaviour on one axis (refusal relative to the base, calibration drop, over-refusal rise,
  coherence drop), damage on the other, with KL below 0.01 not rewarded further. Same KL as kl_first (0.0069),
  more refusals left (0.305). Dropped.
- **v3 to v3.1.** `ablate/guardjudge.py` runs Qwen3Guard-Gen-8B in process and labels each search answer
  "Refusal: Yes/No", the same judge and prompt format as the final evaluation. The phrase rate is still recorded
  beside it. Median over three seeds on Qwen3-8B: refusal 0.191 to 0.043, KL 0.008 to 0.014: real removal, at a
  small KL cost.
- **v3.1 to v3.2.** `rank1_grid()` enqueues 24 fixed trials: rank 1, band kernel, centre 0.50 / 0.60 of depth,
  width 0.15 / 0.25 / 0.40, peak 0.8 / 1.0, plain and protected subspace, both writer kinds, one broadcast
  direction, no norm restore. TPE starts modelling right after the grid. The median stayed about the same; the bad
  seed disappeared (worst seed 0.096 / KL 0.0295 to 0.057 / 0.0129).

On gemma-3-12b only the last step helped (KL 0.092 to 0.038); v3 kl_first changed nothing there, and every
version ends near 25% refusal.

## Engineering for large models

- Hook-mode search (no weight copies, `--edit-mode hook`), optionally with the model held in 4 or 8 bit (`--bnb`).
- CPU stream baker that writes the final model shard by shard and checks that only edited tensors differ.
- Per-trial Optuna journal (`--study-journal`): an interrupted search resumes where it stopped
  ("resuming study: N trials already scored"). Never run two processes on the same run directory.
- `scripts/abliterate_stream.py` + `ablate/streamsearch.py`: for models that do not fit even with hooks, candidate
  plans are scored with one forward pass each on first-token proxies (refusal log-probability, first-token KL).
  This is a proxy only; no reported number comes from it.

## Re-selecting at a different refusal level (matched-refusal points)

The r30 / r60 / r40 / r50 points in the study come from a finished v3.2 search re-selected under a looser
refusal tolerance, with no new trials. Recipe: make a new run directory, copy into it the finished run's
`extract.done`, `extract.json`, `condition.pt`, `subspace_plain.pt`, `subspace_protected.pt` and `study.journal`,
then run `scripts/abliterate.py` with the same config, the same `--trials` and the new `--refusal-tol`. The journal
already holds every trial, so only selection, the final scoring and the card are redone. Then bake and evaluate it
like any other contender. Selection is on the search set, so held-out refusal can land far from the tolerance
(Qwen3-8B: tolerance 0.60 gave 0.351 held-out).

## Things that are known not to work or not to matter

- Matched harmless twins as the extraction contrast (above).
- Share of variance in the top direction as a predictor of damage: correlation -0.08 across 15 runs.
- A random 40-trial probe with a KL ceiling to adapt the search per model: two seeds in three had no good trial
  by trial 40 to 60 on Qwen3-8B, so it would abort runs that end fine. The fixed grid replaced it.
- Stage 5 healing (`--heal-steps`, a self-distillation LoRA) exists in the code but no reported version uses it.
  Distilling towards the untouched teacher can re-teach refusal; `--heal-reproject` projects the merged weights
  again for that reason.
