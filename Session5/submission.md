# V5 Data Mixture & Curriculum Plan

Target capabilities: Codex-style coding/agentic work, controllable reasoning
(low/medium/high/ultra), native Indic fluency (primary differentiator). Every
number below is sized against real supply from the Session 4 inventory
(`Session5/reference/mined-numbers.md`) and tied to the benchmark it is meant
to win. Supply figures cited as `[S]`.

## 1. Pretraining mixture (main run)

| Lane | Share | Tokens @ 5T budget | Benchmarks | Supply check |
|---|---|---|---|---|
| General web + STEM | 34% | 1,700 B | MMLU | 4,837 B available `[S]` — no strain |
| Code | 24% | 1,200 B | LiveCodeBench, Aider | 1,103 B unique `[S]` → **~1.1× repetition** needed |
| Indic | 16% | 800 B | MILU, IndicGenBench | see tier split below — cannot come from verified alone |
| STEM (papers/proofs) | 12% | 600 B | AIME/GPQA (with reason) | peS2o+proof-pile 97 B `[S]` → **~6× repetition**, acceptable for STEM text |
| Reasoning | 6% | 300 B | AIME, GPQA, HLE | open supply ~7 B `[S]` + AON/V4 78 B → **must lean on distillation/synthesis**, see §5 |
| Long-context | 6% | 300 B | long-eval | 100 B repacked supply `[S]` → **~3× repack** of existing long docs, not new tokens |
| Agentic | 2% | 100 B | SWE-bench, tau-bench, BFCL, GAIA, BrowseComp | 0.6 B real trajectories `[S]` → **~99% of this lane is synthetic**, see §5 |

This adopts the course's validated `pretrain` preset (24/2/6/6/16/12/34) rather than
inventing new proportions: it is the one preset in the inventory already checked
against supply and benchmark coverage. Two lanes (agentic, reasoning) are
supply-constrained enough that the share is a **synthesis target**, not a
collection target — stated explicitly rather than hidden behind the headline %.

## 2. Indic tier split (of the 16% / 800B Indic share)

| Tier | Share of Indic slice | Tokens | Source |
|---|---|---|---|
| A — verified native | 40% | 320 B | Sangraha verified (64 B unique `[S]`) → **5× repetition** |
| B — unverified crawl | 25% | 200 B | Sangraha unverified + IndicCorpV2 (~45 B unique `[S]`) → **~4.4× repetition** |
| C — translated | 20% | 160 B | BPCC, Samanantar, IndicTrans2 (~5 B unique `[S]`) → **~32× repetition or fresh MT generation** |
| D — synthetic | 15% | 120 B | LLM-generated Indic (no natural ceiling) |

**Honest accounting:** verified+unverified native supply is only ~109 B tokens
`[S]`, so a 65%-native target (tiers A+B) at 800B×0.65 = 520B requires ~4.8×
repetition of every native token that exists. This is the real cost of making
Indic the primary differentiator — it is not free, and the plan states the
repetition factor rather than a bare percentage. Tier C/D make up the
remaining 35% and are where fresh generation work (translation + synthetic)
must be commissioned, prioritized by the scarcity markers in the inventory.

## 3. Agentic slot (named datasets)

Pointed at inventory tier-A trajectories: SWE-Gym (150M tok), SWE-smith (120M),
OpenHands rollouts (90M), ToolACE (60M), Glaive-v2 (50M), Nexus (30M), xLAM
(25M), Hermes (22M), ToolBench (80M) — **0.6 B tokens total `[S]`**, against a
100 B pretrain target. The gap (>99%) is filled by synthesized multi-step
trajectories in the shape of §6 of the lesson (plan → tool call → observation
→ recovery → answer), generated against the same tool schemas as the real
sets so format is compatible. Loss mask: tool observations are context-only
(grey), planning/calls/final response are trained (green) — never train on
tool-generated text. Benchmarks bought: SWE-bench, tau-bench, BFCL, GAIA,
BrowseComp.

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
the traces it will be trained on.

## 5. Long-context slot

6% / 300B, built from repo-packed code ≥32K tokens (60B `[S]`) and book-length
corpora (40B `[S]`), repacked/concatenated to reach budget rather than newly
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

## 7. Curriculum (stage order + difficulty ladder)

Adopts the course's validated 5-stage curriculum, Indic/reasoning weight
rising monotonically toward the anneal:

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
if nothing upstream consumes the reserve early.

**Mixture-transition safety:** every stage boundary (and the anneal
transition specifically) is warmed up across a multi-billion-token band, not
switched in one step — V4's ungated Hindi-share jump against frozen
embeddings produced a ~150× gradient-norm spike; this plan requires the same
warmup band before any lane's share changes by more than 2 points/step.

## 8. Proxy experiment (required before full-scale trust)

**Scale:** 3B parameters, ~60B tokens, mixture identical in proportion to the
main pretrain preset (§1).

**Hypothesis under test:** the 16%/2%/6% Indic/agentic/reasoning shares with
an 8% protected floor produce measurable gains on MILU/IndicGenBench,
BFCL/tau-bench (a small held-out synthetic-agentic eval), and AIME-mini
respectively, without depressing MMLU by more than 2 points versus a
web-heavy 60% control run at the same scale.

**Metric that confirms/refutes:** ΔMILU, ΔIndicGenBench, ΔBFCL-lite,
ΔAIME-mini, ΔMMLU between the two 3B runs. If any protected lane shows no
measurable delta over its control at 3B, that lane's share is renegotiated
before the 5T run — a data decision is a hypothesis until this experiment
returns numbers.
