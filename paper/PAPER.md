---
title: "Removing refusals with less damage: what worked, what did not, and why"
subtitle: "A controlled study of a searched refusal-direction edit against Heretic and OBLITERATUS on seven open-weight models"
author: "Matrix (NuevaTech LLC)"
date: 2026-11-22
version: "draft 3"
license: "CC BY 4.0 (text), Apache-2.0 (code)"
code: "https://github.com/Limitlessmatrix/abliteration-study"
data: "results/results.json in the code repository (every number in this paper, machine readable)"
---

# Removing refusals with less damage: what worked, what did not, and why

## Abstract

Refusal-direction removal ("abliteration") edits an open-weight language model so it stops refusing, without
retraining. Most published results report how many refusals disappear and say little about what else changed.
We built a searched variant of the edit whose search minimises damage, defined as the KL divergence from the
original model on harmless prompts, subject to a refusal target scored by a safety classifier. We compared it with
two public tools, Heretic 1.4.0 and OBLITERATUS 0.1.4, under one fixed protocol: 282 held-out harmful prompts,
judged refusal, KL on 300 harmless prompts, over-refusal on 300 OR-Bench prompts, and capability on MMLU, ARC and
GSM8K, with paired tests and Holm correction.

What worked: on Qwen3-8B our current version (v3.2) leaves 2.5 to 5.7% refusals at KL 0.011 to 0.013 (three
seeds), down from 13.5% at KL 0.052 for our first version, with capability unchanged. On gemma-3-27b-it it leaves
10.3% at KL 0.049 with MMLU equal to the base, where Heretic leaves 39.7% at KL 0.012. What did not: Heretic keeps
the lowest damage on every model; on gemma-3-12b no setting of our edit gets below about 25% refusal; on
Qwen3.8-27B the edit removes refusals but at a high KL (0.099). Our explanation for the pattern: where refusal is
written by a single direction in a narrow band of layers (Qwen3-8B), it can be removed almost for free; where it is
spread over more directions or rebuilt downstream (gemma-3-12b), a rank-1 edit hits a floor; and tools tuned for
least damage at default settings (Heretic) stop before removing much. The largest lessons are about measurement:
phrase-matched refusal counts, single seeds and search-set scores each produced a wrong headline number during this
study, two of them ours. We also show that the edit is exactly a rank-1 LoRA adapter in the common case, that a
run-time version of the edit on a 321-billion-parameter MoE model keeps MMLU and ARC, and that fine-tuning an edited
model on data with 1% refusal answers brings refusals back.

## 1. Why this study

Open-weight models ship with refusal training. Red teamers, security researchers and people building tools for
them often need a model that will discuss attacks it would otherwise refuse, running on hardware they control.
Abliteration is the cheapest way to get one: find the direction in the model's internal activations that separates
harmful from harmless requests, and remove it from the weights.

Every edit has a cost. A model that stops refusing but also gets worse at everything else is not useful, and a
tool that reports "0 refusals" from a keyword count may be measuring the wrong thing. We wanted to know three
things, each measured on data no search ever saw:

1. Can a searched edit remove more refusals than the public tools at the same or lower damage?
2. Where does it work, where does it fail, and what explains the difference?
3. Which measurement shortcuts mislead, and by how much?

This is research on self-hosted weights for defensive and red-team use.

## 2. Background in one paragraph

Arditi et al. showed that refusal in chat models is mediated largely by one direction in the residual stream:
the difference between mean activations on harmful and harmless prompts. Projecting that direction out of the
matrices that write into the residual stream removes most refusals. Later tools search over which layers to edit
and how strongly. Heretic (1.4.0) runs a multi-objective search over edit strength per layer, trading refusals
(counted by phrase matching on its own prompts) against KL divergence, and picks from its trade-off front.
OBLITERATUS (0.1.4) offers preset edit recipes. Our edit is in the same family: remove k directions (rank k) with
a per-layer strength profile (a "kernel" over depth), and search the kernel's centre, width, strength and rank.

## 3. Evaluation protocol (fixed for every contender)

| Item | Value |
|---|---|
| Harmful prompts | 282 held-out StrongREJECT prompts, never used in any search |
| Over-refusal prompts | 300 OR-Bench-hard prompts (harmless but edgy) |
| Damage | mean KL divergence from the base model, measured on the base model's own greedy answers to 300 held-out Dolly prompts; first-token KL kept separately |
| Refusal judge | Qwen3Guard-Gen-8B, greedy. `refusal` = the judge says the answer refused; `harm_comply` = answered and judged unsafe |
| Capability | lm-eval 0.4.13: MMLU, ARC-Challenge, GSM8K (400 items per task or subtask on the small models; full MMLU on the 27B models) |
| Decoding | the base model's tokenizer and chat template, greedy, for every contender |
| Statistics | paired per-item tests (McNemar for rates, paired bootstrap for KL), Holm correction within each model, 95% intervals |
| Competitors | Heretic 1.4.0 and OBLITERATUS 0.1.4 (advanced and optimized presets), from pinned, audited releases, at default settings |

Rules we fixed before looking at results: only held-out, judged numbers are quoted as findings; search-set numbers
are labelled as such; a claim of "less damage than Heretic" needs our KL below Heretic's at a held-out refusal equal
to or below Heretic's.

For Heretic we evaluate the trial with the fewest refusals on its own trade-off front (KL up to 0.5): its
strongest removal at defaults, not a cautious pick.

## 4. The method and how it changed

All versions share one idea: extract the directions that separate harmful from harmless prompts, then remove them
from the weights that write into the residual stream, with a kernel over depth that sets how strongly each layer is
edited. The versions differ in what the search optimises and how it scores refusal while searching.

| Version | What it is | What changed, and why |
|---|---|---|
| v2 | protected refusal subspace, rank up to 4, band or gaussian kernel; 32 trials; objective = refusal + KL weighted 1:1; refusal counted by phrase matching | Starting point. Removed refusals, at 9 times Heretic's KL. |
| v3 kl_first | objective becomes "minimise KL subject to refusal and calibration tolerances"; 150 trials; edits applied through hooks (no weight copies); resumable trial journal | The 1:1 objective was buying refusal removal with damage. KL fell 7 times, but judged refusal rose: the phrase counter told the search refusal was 0.000 when the judge said 0.213. |
| v3 pareto | two objectives in the style of Heretic, KL target 0.01 | Tried as an alternative. Same KL as kl_first, more refusals left. Dropped. |
| v3.1 kljudge | v3 kl_first with refusal scored inside the search by the same judge used for evaluation | With the judge in the loop the search stopped optimising a number that was not real. |
| v3.2 kljgrid | v3.1 with the first 24 trials fixed to a grid of narrow rank-1 band edits, then the sampler continues | One seed in three never found the narrow band and ended at 3 times the KL. The grid gives every run the same starting map. |

Engineering that made the large models possible: search through forward hooks (no weight copies), a CPU "stream
baker" that writes the final model shard by shard and verifies that only the intended tensors differ from the
base, and a per-trial journal so an interrupted search resumes.

Effect of each step on Qwen3-8B (held-out, medians over three seeds where available):

| Step | Effect |
|---|---|
| v2 to v3 kl_first | KL 0.052 to 0.008, judged refusal 0.135 to 0.191: the search had minimised a refusal score that was not real |
| v3 to v3.1 (judge in the search) | refusal 0.191 to 0.043, KL 0.008 to 0.014: real removal, small KL cost |
| v3.1 to v3.2 (grid start) | median about the same; the bad seed disappears (worst seed 0.096 / 0.0295 to 0.057 / 0.0129) |

## 5. Results

All numbers held-out and judged unless marked "search set". Refusal and over-refusal are rates; KL is mean
per-token KL from the base.

### 5.1 Qwen3-8B (three seeds for the searched versions)

| Contender | Refusal | Answered unsafe | Over-refusal | KL | MMLU | ARC | GSM8K |
|---|---|---|---|---|---|---|---|
| base | 0.954 | 0.004 | 0.370 | 0 | 0.757 | 0.560 | 0.892 |
| ours v2 | 0.135 | 0.777 | 0.003 | 0.0521 | 0.758 | 0.560 | 0.905 |
| ours v3 kl_first (seeds 0 / 1 / 2) | 0.213 / 0.163 / 0.191 | 0.695 / 0.723 / 0.706 | 0.000 / 0.007 / 0.003 | 0.0071 / 0.0147 / 0.0079 | 0.759 / 0.756 / 0.758 | 0.560 / 0.542 / 0.565 | 0.895 / 0.912 / 0.900 |
| ours v3.1 kljudge (seeds 0 / 1 / 2) | 0.032 / 0.096 / 0.043 | 0.879 / 0.812 / 0.862 | 0.000 / 0.007 / 0.013 | 0.0121 / 0.0295 / 0.0136 | 0.759 / 0.754 / 0.757 | 0.550 / 0.557 / 0.552 | 0.902 / 0.905 / 0.927 |
| **ours v3.2 kljgrid (seeds 0 / 1 / 2)** | **0.025 / 0.057 / 0.053** | 0.887 / 0.872 / 0.876 | 0.003 / 0.003 / 0.000 | **0.0129 / 0.0109 / 0.0112** | 0.760 / 0.758 / 0.758 | 0.555 / 0.547 / 0.552 | 0.902 / 0.895 / 0.902 |
| ours v3.2 re-selected at refusal tolerance 0.30 (r30) | 0.117 | 0.794 | 0.003 | 0.0091 | 0.759 | 0.555 | 0.895 |
| ours v3.2 re-selected at refusal tolerance 0.60 (r60) | 0.351 | 0.546 | 0.003 | 0.0070 | 0.757 | 0.557 | 0.895 |
| Heretic 1.4.0 | 0.610 | 0.273 | 0.050 | 0.0057 | 0.757 | 0.557 | 0.897 |
| OBLITERATUS optimized | 0.876 | 0.053 | 0.110 | 0.1157 | 0.746 | 0.547 | 0.870 |
| OBLITERATUS advanced | 0.716 | 0.188 | 0.113 | 0.2420 | 0.747 | 0.525 | 0.880 |

Ours against Heretic, paired (ours minus Heretic, Holm): r60 refusal -0.259 [-0.316, -0.202] significant, KL
+0.0013 [+0.0006, +0.0020] p 0.074 not significant after correction; r30 refusal -0.493 significant, KL +0.0034
significant; v3.2 refusal -0.55 to -0.59 significant, KL +0.005 to +0.007 significant. Our trade-off curve on this
model (refusal / KL): 0.025 / 0.0129, 0.117 / 0.0091, 0.213 / 0.0071, 0.351 / 0.0070. It flattens at KL about
0.0070; we have no point below Heretic's 0.0057.

### 5.2 gemma-3-12b-it (one seed per row)

| Contender | Refusal | Answered unsafe | Over-refusal | KL | First-token KL | MMLU | ARC | GSM8K |
|---|---|---|---|---|---|---|---|---|
| base | 0.954 | 0.011 | 0.173 | 0 | 0 | 0.738 | 0.610 | 0.882 |
| ours v2 | 0.259 | 0.660 | 0.000 | 0.0913 | 0.398 | 0.733 | 0.588 | 0.882 |
| ours v3 kl_first | 0.270 | 0.638 | 0.003 | 0.0920 | 0.360 | 0.733 | 0.583 | 0.880 |
| **ours v3.2 kljgrid** | **0.252** | 0.667 | 0.010 | **0.0381** | 0.320 | 0.736 | 0.615 | 0.877 |
| ours v3.2 r40 | 0.333 | 0.589 | 0.007 | 0.0273 | 0.209 | 0.736 | 0.610 | 0.885 |
| ours v3.2 r50 | 0.592 | 0.326 | 0.027 | 0.0206 | 0.121 | 0.736 | 0.608 | 0.890 |
| ours v3.2, matched-pairs extraction | 0.457 | | | 0.0399 | | | | |
| Heretic 1.4.0 | 0.496 | 0.418 | 0.017 | 0.0269 | 0.127 | 0.739 | 0.610 | 0.895 |

OBLITERATUS does not support this architecture. At equal KL (r40 0.0273 against Heretic 0.0269, not different) ours
leaves 33% refusals against 50% (significant); Heretic changes the first token of answers less (0.127 against 0.209,
significant). At lower KL (r50, 0.0206, significant) ours leaves 59%, not significantly more than Heretic.

### 5.3 Qwen3-32B (older v2-style search only)

| Contender | Refusal | Over-refusal | KL | First-token KL | MMLU | ARC |
|---|---|---|---|---|---|---|
| base | 0.965 | 0.313 | 0 | 0 | 0.832 | 0.573 |
| ours (v2-style) | 0.213 | 0.007 | 0.0186 | 0.110 | 0.829 | 0.562 |
| Heretic 1.4.0 | 0.298 | 0.003 | 0.0278 | 0.041 | 0.834 | 0.578 |
| OBLITERATUS advanced | 0.901 | 0.273 | 0.1939 | 1.109 | 0.827 | 0.557 |
| OBLITERATUS optimized | 0.897 | 0.083 | 0.2570 | 0.354 | 0.822 | 0.593 |

Ours is lower on KL (significant) and level on refusal (p 0.055). GSM8K is left out: the base scores 0.605 and the
edited models about 0.69, an answer-format artefact for this model.

### 5.4 The 27B pair (one seed per row, full MMLU)

| Model / contender | Refusal | Answered unsafe | Over-refusal | KL | First-token KL | MMLU | ARC | GSM8K |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B base | 0.989 | 0.000 | 0.873 | 0 | 0 | 0.851 | 0.578 | 0.730 |
| ours v2-style | 0.521 | 0.326 | 0.037 | 0.0683 | 0.236 | 0.831 | 0.590 | 0.708 |
| **ours v3.2** | **0.117** | 0.766 | 0.003 | 0.0987 | 0.266 | 0.833 | 0.573 | 0.777 |
| ours v3.2 re-selected (r10) | 0.255 | 0.635 | 0.017 | 0.0347 | 0.168 | 0.843 | 0.583 | 0.765 |
| Heretic 1.4.0 | 0.979 | 0.000 | 0.603 | 0.0036 | 0.030 | 0.851 | 0.575 | 0.705 |
| gemma-3-27b-it base | 0.933 | 0.025 | 0.137 | 0 | 0 | 0.783 | 0.590 | 0.915 |
| ours v2-style | 0.305 | 0.617 | 0.003 | 0.1354 | 0.103 | 0.769 | 0.600 | 0.910 |
| **ours v3.2** | **0.103** | 0.819 | 0.007 | **0.0488** | 0.098 | **0.783** | 0.593 | 0.905 |
| Heretic 1.4.0 | 0.397 | 0.528 | 0.010 | 0.0124 | 0.107 | 0.784 | 0.595 | 0.912 |

gemma-3-27b-it: v3.2 against the old edit, refusal -0.20 and KL -0.087 (both significant), MMLU +0.014
(significant) and equal to the base (difference -0.0004, not significant). Against Heretic: refusal -0.29
(significant), KL +0.036 (significant), first-token KL not different.

Qwen3.8-27B: v3.2 removes refusals (0.117) but costs KL 0.099 and 1.8 points of MMLU (significant). This edit used
column norm restore, a variant that rescales weights after the projection; the re-selected r10 point trades 14 points
of refusal for a third of the KL and most of the MMLU back. Heretic at defaults did nothing on this model: 97.9%
refusal and zero unsafe answers, though over-refusal fell from 0.873 to 0.603.

### 5.5 Qwen2.5-72B (search set only)

60 trials on three A100-80GB cards. Selected edit: rank 1, band centred at 0.70 of depth, width 0.52. Search set (63
prompts): judged refusal 0.111, KL 0.054, over-refusal 0.019. The baked model was verified (130 of 963 tensors
differ, the rest bit-identical). No held-out evaluation and no competitor, so no comparative claim.

### 5.6 Run-time edit of a 321B MoE: GLM-5.3-Flash through Project Maya

Project Maya (MIT, pinned commit 56b5216) runs GLM-5.3-Flash from a 73 GB 1-bit quantisation on one A100-80GB and
ships one refusal direction that it projects out of every residual write at run time, with a scale. We measured the
shipped direction under our protocol (KL on 100 harmless prompts, because each KL measurement needs an engine
start; capability generative through the API, so compare rows with each other only).

| Setting | Judged refusal | Phrase-count refusal | Over-refusal | KL | MMLU | ARC | GSM8K |
|---|---|---|---|---|---|---|---|
| base | 0.989 | 0.982 | 0.290 | 0 | 0.836 | 0.945 | 0.958 |
| scale 0.5 | 0.872 | 0.706 | | 0.029 | | | |
| scale 0.75 | 0.606 | 0.287 | | 0.034 | | | |
| scale 1.0 (default) | 0.238 | 0.046 | 0.003 | 0.043 | 0.835 | 0.949 | 0.934 |
| scale 1.25 | 0.089 | 0.007 | | 0.053 | | | |
| scale 1.5 | 0.018 | 0.004 | | 0.065 | 0.828 | 0.946 | 0.936 |
| scale 2.0 | 0.007 | 0.004 | | 0.101 | | | |

At the default scale the judge finds 24% refusals where a phrase count finds 5%. Scale 1.5 reaches 1.8% at KL
0.065 with MMLU and ARC unchanged and GSM8K 2.2 points lower (500 items; not tested for significance). The damage
floor is high: scale 0.5 removes almost nothing and already costs KL 0.029, which is what one direction applied to
all 90 residual writes should do. One run per row, 1-bit quantisation only.

### 5.7 The edit is a LoRA adapter

Every v3.2 edit without norm restore has the form W' = W - a Q^T Q W, which is exactly a rank-k LoRA (B = -a Q^T,
A = Q W). We exported the Qwen3-8B v3.2 edit as a 2 MB rank-1 adapter, verified that base plus adapter reproduces the
baked weights after rounding, and evaluated both in llama.cpp. Judged refusal, baked against adapter: within noise
at Q4 and Q8 (McNemar p 0.69 and 0.73). First-token KL at Q4 failed our pre-written rule (0.0140 against a noise
floor of 0.0030), but both Q4 builds were equally far from the high-precision baked model (0.062 against 0.059,
interval includes zero), so the gap is two different quantisation errors. That reading was chosen after seeing the
data and needs a second model. Column norm restore is not low rank and cannot be exported exactly. Practical note:
llama.cpp and PEFT load the adapter; Ollama 0.40 refuses adapters.

## 6. What worked, what did not, and why

**Worked: Qwen3-8B, gemma-3-27b-it, and the measurement discipline.**
On Qwen3-8B the search converges across seeds on the same kind of edit: one direction (rank 1) in a band of layers
centred at 0.57 to 0.61 of depth and 0.2 to 0.3 of depth wide. The trade-off curve is nearly flat: from 65% to 97% of refusals removed, KL stays
between 0.0070 and 0.0129. Our reading is that refusal in this model behaves like one feature written in
a narrow range of layers, so removing it there touches little else. gemma-3-27b-it behaved similarly well (10%
refusal, MMLU equal to the base). The mechanism is inferred from what the search picks, not tested by activation
patching.

**Did not work: the gemma-3-12b floor.** Three versions end at 25 to 27% refusal. No setting of the search moved
the floor; v3.2 only lowered the damage at that level (KL 0.091 to 0.038). A fixed grid of narrow rank-1 edits found
nothing useful (best KL 0.10 at up to 50% refusal). Changing the contrast used for extraction to matched pairs made it
worse (0.457). Since the search space does not move the floor, the limit is the edit family or the extracted
direction, not the search. The likely reasons, untested: refusal in this model is carried by more than the few
directions we extract, or later layers rebuild it after the edit. The 27B Gemma does not show the floor, so it is not
a property of the Gemma family as a whole.

**Did not work well: Qwen3.8-27B.** The edit removes refusals but at the highest KL in the study. It is the only
v3.2 winner that used column norm restore, and the search's own trade-off front had a cliff: a point at the same
search-set refusal with 2.7 times less KL. We picked the winner by a strict refusal tolerance; the r10 re-selection
shows the cheaper region exists. The lesson is to look at the whole front before choosing.

**Why Heretic always has the lowest KL.** Heretic picks from a front built with its own phrase-counted refusals on
its own prompts. Its count says its Qwen3-8B build refuses 11%; our judge on held-out prompts says 61%. When the
refusal signal under-counts, a search that also minimises KL stops early, at a point that looks finished to it and
is gentle because it removed little. On Qwen3.8-27B it removed nothing at all. This is not a flaw in Heretic's
design so much as a consequence of what it measures.

**Why OBLITERATUS did worst.** At default presets it removed few refusals (72 to 90% left) and did the most damage
(KL 0.12 to 0.26). Its presets were not designed for these models; tuning may change this, and we did not tune it.

**Why the run-time edit has a high damage floor.** Projecting one direction out of every residual write is a blunt
instrument: the edit touches layers where the direction carries other information. Our searched edits restrict the
edit to a band of layers for exactly this reason. A per-layer fit for GLM is the obvious next step.

## 7. Measurement lessons

1. **Judge refusal; do not pattern-match it.** Our search reported 0.000 refusal by phrase matching on a model the
   judge scored at 0.213. Maya's default looks like 5% by phrases and is 24% by the judge. Heretic's Qwen3-8B build
   is 11% by its count and 61% by ours. Models learn to refuse in words the list does not contain.
2. **Report medians over seeds.** With the same recipe, one seed in three landed at 3 times the KL.
3. **The search's score of its winner is biased low.** Picking the minimum of many noisy estimates does that: on
   every v3.2 run the selected edit re-scored worse at the end than inside the search (0.048 to 0.079, 0.111 to
   0.254 on 63 prompts; 0.111 to 0.206 on gemma-3-27b).
4. **Compare at matched refusal, and expect to miss the match.** A tool that removes little always looks gentle.
   Aiming our search at Heretic's refusal level, the 63-prompt search set mispredicted held-out refusal on Qwen3-8B by
   a wide margin (0.619 predicted, 0.351 measured).
5. **The gain is model-specific.** Check each model's trade-off curve before claiming a general method.
6. **Things we got wrong and corrected:** the phrase-matched zeros; a claim that the share of variance in the top
   direction predicts damage (correlation -0.08 over 15 runs: it does not); a claim that the judge made the search
   repeatable (the third seed disproved it); a probe design that testing showed could not work.
7. **Small details that move numbers:** generation batch size changes about 2% of GSM8K items (keep it fixed per
   model); KL was bit-reproducible across A100 nodes; GSM8K was not a usable measure on Qwen3-32B as we ran it.

## 8. Fine-tuning can put refusals back

We fine-tuned an edited Qwen3.8-27B with a LoRA (rank 64, 289 steps) on a security-task mix. The result refused 6 of
6 mild direct requests the edited base had answered. The cause was in the data: 526 of the 53,893 rows (1.0%) in one
question-answer source were refusal answers from two public datasets, in the wording the fine-tuned model then used.
The fine-tune's own check missed them because it matched different phrases (lesson 1 again). After removing those
rows and rescanning, a second fine-tune of the gemma-3-27b edit kept zero refusals on our comply prompts, while the
Qwen3.8-27B one still refused 2 of 35; the source of those is not yet found. Scan training data for refusal answers
with a judge before fine-tuning an edited model, and judge the result. The fine-tuned models are not released.

## 9. Threats to validity

- Competitors ran at default settings. OBLITERATUS in particular may do better with tuning.
- The judge is treated as ground truth; no human-labelled sample has checked it. The Heretic Qwen3-8B row (judge 61%,
  Heretic's count 11%) is where a human read matters most.
- Seeds: three for the searched versions on Qwen3-8B, one everywhere else, including every Gemma and 27B row and
  both competitors. Matched-refusal points are re-selections from one search.
- All v3.x searches on the small models reuse the v2 extraction; the 27B searches used a fresh extraction.
- Hardware differed between rows (A100, V100, A6000); two evaluations resumed after preemption. KL was shown
  reproducible across nodes; judged refusal was not separately checked.
- The 72B result is search-set only. The GLM rows use a 1-bit quantisation and 100 KL prompts.
- KL on harmless prompts and three benchmarks do not measure everything: long-form quality, tool use and
  multilingual behaviour are untested.
- The adapter reading (5.7) was chosen after seeing the data. The fine-tuning observation (8) uses small probe sets.

## 10. What we would do next

1. Show both tools as curves: Heretic's full trade-off front next to ours, with seeds for the matched points.
2. Diagnose the gemma-3-12b floor: does refusal come back downstream after the edit, how many directions carry it,
   which layers write it.
3. New edit families for the floor: an edit of the instruction-tuning delta along the refusal directions, a learned
   low-rank patch with a KL penalty, a short healing LoRA after a stronger edit.
4. Fix the selection bias: confirm the top candidates on a second prompt set before choosing.
5. Held-out evaluation of the 72B; a per-layer direction fit for GLM-5.3-Flash.
6. A human-labelled check of the judge.

## 11. Release and responsible use

We release the code (Apache-2.0), every number in this paper as machine-readable data, and the v3.2 edited models
for Qwen3-8B, gemma-3-12b-it, gemma-3-27b-it and Qwen3.8-27B as GGUF files with cards that state their measured
refusal, damage, and where a competitor does better. We do not release the fine-tuned security models or the
training data built for them.

The edited models answer requests their base models refuse. They are meant for red teaming, security research and
teaching on hardware the user controls. Removing refusals does not add knowledge the base model lacks; it removes a
gate. Our results also carry a message for model developers: on several models one rank-1 edit removed most refusals
at a KL near 0.01, so refusal training alone is a thin layer of protection for open weights.

## Acknowledgments

Compute for this study was provided by the National Research Platform (NRP Nautilus). The NRP did not review or
endorse this work. We thank the authors of Heretic, OBLITERATUS, Project Maya, the StrongREJECT, OR-Bench and Dolly
datasets, Qwen3Guard, and lm-evaluation-harness.

## How to use this paper and its data

- Humans: sections 6 and 7 are the findings; section 5 holds every table.
- Programs and AI agents: `results/results.json` in the code repository holds every row of section 5 with field
  names `model`, `contender`, `seed`, `refusal`, `harm_comply`, `overrefusal`, `kl`, `kl_first`, `mmlu`, `arc`,
  `gsm8k`, `split` ("heldout" or "search"), and `notes`. Paired tests are in `results/paired_tests.csv`.
