# V5 Data Mixture & Curriculum Plan

Target capabilities: Codex-style coding/agentic work, controllable reasoning
(low/medium/high/ultra), native Indic fluency (primary differentiator).
Token-supply figures marked `[S]` throughout come from a dataset-supply
survey I compiled across the specific datasets named in each section, sized
in tokens rather than samples.

## 1. Pretraining mixture (main run)

| Lane | Share | Tokens @ 3T budget | Benchmarks | Supply check |
|---|---|---|---|---|
| General web + STEM | 34% | 1,020 B | MMLU | 4,837 B available `[S]` — no strain |
| Code | 24% | 720 B | LiveCodeBench, Aider | 1,103 B unique `[S]` → **fits within unique supply, no repetition needed** |
| Indic | 16% | 480 B | MILU, IndicGenBench | see tier split below — cannot come from verified alone |
| STEM (papers/proofs) | 12% | 360 B | AIME/GPQA (with reason) | peS2o+proof-pile 97 B `[S]` → **~3.7× repetition**, acceptable for STEM text |
| Reasoning | 6% | 180 B | AIME, GPQA, HLE | open supply ~7 B `[S]` + AON/V4 78 B → **~2.1× repetition, still leans on distillation/synthesis** for the hardest traces, see §4 |
| Long-context | 6% | 180 B | long-eval | 100 B repacked supply `[S]` → **~1.8× repack** of existing long docs, not new tokens |
| Agentic | 2% | 60 B | SWE-bench, tau-bench, BFCL, GAIA, BrowseComp | 0.6 B real trajectories `[S]` → **~99% of this lane is synthetic**, see §3 |

I'm keeping general web the largest lane (34%) because it's the most abundant,
cheapest data to acquire, with code next (24%) since it's likewise abundant
and buys the coding benchmarks directly. The harder-to-source capabilities —
Indic, reasoning, agentic — get smaller headline shares here precisely
because they're scarce, and are protected instead through the floor in §6
and the anneal reserve in §7 rather than through raw pretraining share.
Two lanes (agentic, reasoning) are supply-constrained enough that the share
is a **synthesis target**, not a collection target — stated explicitly
rather than hidden behind the headline %.
Every repetition figure in the Supply check column stays at or under 4×,
following Muennighoff et al.'s finding that repeated data teaches the model
almost nothing past that point (arXiv:2305.16264, worked through in detail in
§2) — the synthetic and freshly-generated tokens (agentic, most of reasoning,
Indic tier D) aren't bound by that cap since they're new tokens, not repeats.

## 2. Indic tier split (of the 16% / 480B Indic share)

| Tier | Share of Indic slice | Tokens | Source |
|---|---|---|---|
| A — verified native | 40% | 192 B | Sangraha verified (64 B unique `[S]`) → **3.0× repetition** |
| B — unverified crawl | 25% | 120 B | Sangraha unverified + IndicCorpV2 (~45 B unique `[S]`) → **~2.7× repetition** |
| C — translated | 20% | 96 B | BPCC, Samanantar, IndicTrans2 (~5 B unique `[S]`, capped at 4× = 20B) + **fresh MT generation for the remaining ~76B** |
| D — synthetic | 15% | 72 B | LLM-generated Indic (no natural ceiling) |

**Honest accounting:** verified+unverified native supply is only ~109 B tokens
`[S]`, so a 65%-native target (tiers A+B) at 480B×0.65 = 312B needs ~2.9×
repetition of every native token that exists — inside the safe range, but
still a real cost worth stating rather than a bare percentage.

I'm capping every tier's *effective* repetition at ~4×, following Muennighoff
et al., *Scaling Data-Constrained Language Models* (arXiv:2305.16264): up to
~4 epochs of repeated data costs next to nothing versus unique tokens, but
past that the marginal value of a repeated token decays toward zero. At this
3T budget both A (3.0×) and B (2.7×) sit comfortably inside that cap on
their own — I don't need to shrink either tier the way I did at a 5T budget,
where the same naive split pushed Tier C to 32× repetition and forced a
reshuffle between B and C. Tier C is still the constrained one: its ~5B
unique translated corpus caps out around 20B under the 4× rule, so the
remaining ~76B of its 96B target has to come from commissioning fresh MT
generation, not from repeating the same sentences past the point they teach
the model anything new. Tier D stays the flexible lane since synthetic
generation has no repetition ceiling by construction.

## 3. Agentic slot (named datasets)

Pointed at named tier-A trajectory datasets: SWE-Gym (150M tok), SWE-smith (120M),
OpenHands rollouts (90M), ToolACE (60M), Glaive-v2 (50M), Nexus (30M), xLAM
(25M), Hermes (22M), ToolBench (80M) — **0.6 B tokens total `[S]`**, against a
60 B pretrain target. The gap (>99%) is filled by synthesized multi-step
trajectories shaped like a real multi-step task (plan → tool call →
observation → recovery → answer), generated against the same tool schemas as the real
sets so format is compatible. I want these generated trajectory-first rather
than task-first: execute real tools/APIs, keep only trajectories that
actually succeed, then generate or backtranslate the task description from
the verified trajectory. This follows APIGen-MT and TaskCraft, and mirrors
Trajectory2Task and Firefly, all 2025 work converging on the same fix —
asking a model to invent both a task and its solution at once is exactly
where synthetic tool-use data goes wrong, because nothing forces the
"solution" to be real. Starting from execution and working backward is what
makes the synthetic 99% trustworthy rather than merely plausible-looking.
Loss mask: tool observations are context-only (grey), planning/calls/final
response are trained (green) — never train on tool-generated text.
Benchmarks bought: SWE-bench, tau-bench, BFCL, GAIA, BrowseComp.

## 4. Reasoning slot (trace-length bands)

Reserved for the post-pretrain reasoning-training stage (Sessions 17-18), not
mixed flat into pretraining. Three trace-length bands, each needing real
examples at every reasoning-effort level:

| Band | Length | Effort levels covered | Source |
|---|---|---|---|
| Short | 1-2 steps | low | direct-answer math/code (OpenMathReasoning subset) |
| Medium | multi-step, no backtrack | medium | OpenThoughts2, NuminaMath |
| Long | verifies + corrects intermediate steps | high/ultra | OpenR1-Math (R1-distilled), AON/V4 |

Concrete example per band (same problem, "a train leaves station A..."):
- **Low:** states the formula, plugs in numbers, one-line answer.
- **Medium:** sets up the equation, solves, states the answer with one check.
- **High/ultra:** explores two solution paths, verifies the arithmetic,
  restates the answer with a sanity check against a boundary case.

Benchmarks: AIME, GPQA, HLE. RLVR (verifiable-reward RL) is the later stage
that teaches the model to *obey* the effort setting — this slot only supplies
the traces it will be trained on. For that later stage I'd follow ThinkDial
(arXiv:2508.18773): budget-mode SFT on the length-banded traces above,
followed by two-phase budget-aware RL with adaptive reward shaping, which is
the concrete recipe that turns "traces of different lengths exist in the
mixture" into "the model reliably matches its output length to the
requested effort." BudgetThinker (arXiv:2508.17196) is the fallback if
budget-mode SFT alone underperforms — it reminds the model of its remaining
token budget with control tokens during generation rather than relying on
the initial prompt alone.

## 5. Long-context slot

6% / 180B, built from repo-packed code ≥32K tokens (60B `[S]`) and book-length
corpora (40B `[S]`), repacked/concatenated to reach budget (~1.8× the 100B raw
supply, under the 4× cap) rather than newly
collected — introduced after the general/reasoning stages so the model
already knows how to reason before learning to hold it over long spans.
Benchmark: long-eval.

## 6. Protected always-on floor

**8% of every OPUS iteration**, extended from V4's Indic-only floor to cover
**Indic + agentic + reasoning** — the three lanes an English-heavy selector
proxy would starve first (V4 evidence: unprotected Indic collapsed toward
zero under English-heavy scoring). OPUS keep-rate stays at the validated 40%
(~6× effective-token multiplier, few-% compute overhead `[S]`) for the
remaining 92% of every iteration.

I know this cuts against *Revisiting Multilingual Data Mixtures in Language
Model Pretraining* (arXiv:2504.04152), which argues language-ordering/balancing offers no
clear benefit over a well-mixed corpus and that the investment is better
spent scaling and cleaning low-resource data than on protection machinery.
I'm not dismissing that finding, but it wasn't tested against an active
selector — my exact failure mode (unprotected Indic collapsing under an
English-heavy OPUS proxy) is causal evidence from our own V4 run, not a
hypothetical. Scaling and cleaning Indic data (Sessions 3-4) and protecting
it from the selector (this section) aren't substitutes for each other; I'm
doing both.

## 7. Curriculum (stage order + difficulty ladder)

I'm using a 5-stage curriculum, with Indic/reasoning weight rising
monotonically toward the anneal. This broad-to-specific shape matches
what *Beyond Random Sampling: Efficient Language Model Pretraining via
Curriculum Learning* (arXiv:2506.11300) finds works in practice, and the
same multi-stage pattern (general web first, quality/specialty data later)
is what OLMo 2, Phi-4 and LongCat-Flash actually shipped with — I'm not
inventing an untested ordering.

| Stage | gen/code/reason/lctx/indic/stem |
|---|---|
| Seed | 55/15/3/2/15/10 |
| General | 45/20/6/3/16/10 |
| Reasoning | 25/28/18/5/14/10 |
| Long-context | 18/30/16/18/12/6 |
| **Anneal (reserve)** | 8/22/20/10/**30**/10 |

Within each stage, difficulty climbs B0→B5 (Nursery → Grade-school →
High-school → Undergraduate → Graduate → Research/PhD), so simple material is
seen before hard material at every stage rather than once globally.

**Anneal reserve (§ what is held back):** Tier-A verified-Indic and Tier-A
long-reasoning traces are excluded from ordinary OPUS sampling throughout
pretraining and spent only in the final low-LR anneal, where Indic rises to
30% — this is the single highest-leverage lever in the plan and it only works
if nothing upstream consumes the reserve early. MiniCPM's WSD scheduler
(arXiv:2404.06395, arXiv:2506.07900) is direct precedent for the mechanism
itself: a stable phase followed by a decay phase trained on held-back
high-quality data.

One risk I'd otherwise have missed: *How Learning Rate Decay Wastes Your Best
Data in Curriculum-Based LLM Pretraining* (arXiv:2511.18903) shows that
best-data-last curricula can conflict with a decaying LR schedule — if the
reserve arrives while LR is already deep in its terminal decay, its gradient
contribution is underweighted and the reserve is partly wasted regardless of
how carefully it was protected. So I'm sequencing the anneal stage to begin
while LR is still a meaningful fraction of peak, not in the decay tail — the
low-LR cooldown and the mixture shift onto the reserve happen together, not
one after the other.

**Mixture-transition safety:** every stage boundary (and the anneal
transition specifically) is warmed up across a multi-billion-token band, not
switched in one step — V4's ungated Hindi-share jump against frozen
embeddings produced a ~150× gradient-norm spike; this plan requires the same
warmup band before any lane's share changes by more than 2 points/step.

## 8. Proxy experiment (required before full-scale trust)

I'm running this in two steps rather than a single A/B, because a single
comparison only tells me my proposed mixture beats one alternative, not that
it's near the best mixture reachable at this budget.

**Step 1 — mixture sweep (RegMix-style):** train ~12-16 tiny proxy models
(~150M-1B params, a few billion tokens each) at mixtures sampled around my
proposed shares in §1, then fit a regression predicting validation loss and
the four target benchmarks from the mixture proportions. This is RegMix
(arXiv:2407.01492): it matches DoReMi's proxy-based reweighting
(arXiv:2305.10429) at roughly 10% of the compute, and Data Mixing Laws
(arXiv:2403.16952) plus Scaling Laws for Optimal Data Mixtures
(arXiv:2507.09404) are what justify extrapolating the fitted relationship
from these tiny runs up toward the full run rather than treating each scale
as an unrelated experiment. I use the fit to check my §1 shares actually sit
near the predicted optimum for the four target benchmarks, and move them if
they don't, before spending a 3B run confirming a mixture the sweep already
says is off.

**Step 2 — confirmation run.** 3B parameters, ~60B tokens, mixture identical
in proportion to the (possibly adjusted) shares from Step 1.

**Hypothesis under test:** the Indic/agentic/reasoning shares with an 8%
protected floor produce measurable gains on MILU/IndicGenBench, BFCL/tau-bench
(a small held-out synthetic-agentic eval), and AIME-mini respectively, without
depressing MMLU by more than 2 points versus a web-heavy 60% control run at
the same scale.

**Metric that confirms/refutes:** ΔMILU, ΔIndicGenBench, ΔBFCL-lite,
ΔAIME-mini, ΔMMLU between the two 3B runs. If any protected lane shows no
measurable delta over its control at 3B, that lane's share is renegotiated
before the 3T run — a data decision is a hypothesis until this experiment
returns numbers.

## References

- Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* — arXiv:2305.10429
- Liu et al., *RegMix: Data Mixture as Regression for Language Model Pre-training* — arXiv:2407.01492
- Ye et al., *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance* — arXiv:2403.16952
- *Scaling Laws for Optimal Data Mixtures* — arXiv:2507.09404
- Muennighoff et al., *Scaling Data-Constrained Language Models* — arXiv:2305.16264
- *Revisiting Multilingual Data Mixtures in Language Model Pretraining* — arXiv:2504.04152
- Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* — arXiv:2404.06395
- *MiniCPM4: Ultra-Efficient LLMs on End Devices* — arXiv:2506.07900
- *How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining* — arXiv:2511.18903
- *Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning* — arXiv:2506.11300
- *ThinkDial: An Open Recipe for Controlling Reasoning Effort in Large Language Models* — arXiv:2508.18773
- *BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens* — arXiv:2508.17196
