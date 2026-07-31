# Session 5 — mined reference numbers (from the course widgets)

Everything here is extracted from `reference/widgets/*.html`. These are the anchors
the plan must be sized against.

## Real supply per lane (tokens) — dataset inventory (widget_9)

| Lane | Total tokens | Notes |
|------|-------------|-------|
| General web + STEM | **4,837 B** | DCLM 2600, FineWeb-Edu 1300, D2 627, D1 164, proof-pile 55, D4-STEM 49, peS2o 42 |
| Code | **1,103 B** | Stack v2 900, D3/V4 199, CommitPack 4 |
| Indic | **275.9 B** (see tier split) | Sangraha verified 64 / unverified 24 / synthetic 162; IndicCorpV2 20.9; BPCC 3; Samanantar 2 |
| Long-context | **100 B** | repo-packed 60, book-length 40 (repacked, not new) |
| Reasoning + math | **85.1 B** | AON/V4 78 is the bulk; open: OpenThoughts 3, OpenMath 2, OpenR1 1.6, Numina 0.5 (~7B non-V4) |
| Agentic | **0.6 B** | SWE-Gym 150M, SWE-smith 120M, OpenHands 90M, ToolBench 80M, ToolACE 60M, Glaive 50M, Nexus 30M, xLAM 25M, Hermes 22M |

Composer's SUPPLY_T (T tokens): code 1.1, agentic **0.00063**, reason 0.085, longctx 0.1, indic 0.276, stem 0.25, web 4.5.

**Scarcity ranking:** agentic (0.6B) ≪ open-reasoning (~7B) < verified-Indic (64B) < long-ctx (100B) ≪ code/web.

## Indic tier split — share of the Indic slice (widget_1 TIERS, Session 3)

A verified native **40%** · B unverified crawl **25%** · C translated **20%** · D synthetic **15%**.
Real native supply ≈ 64B verified + ~45B unverified ≈ **~109B unique native** (matches S3 "90–110B").

## Benchmark → lane map (widget_1 BENCH)

- Code → LiveCodeBench, Aider
- Agentic/tool-use → SWE-bench, tau-bench, BFCL, GAIA, BrowseComp
- Reasoning+STEM → AIME, GPQA (reason+stem), HLE (reason)
- Long-context → long-eval
- Indic → MILU, IndicGenBench
- General web → MMLU

## Course preset mixtures (widget_1 PRESETS) — order: code, agentic, reason, longctx, indic, stem, web

- **pretrain (main):** 24 / 2 / 6 / 6 / 16 / 12 / 34
- **anneal (final):** 20 / 8 / 18 / 8 / 28 / 10 / 8
- naive (bad): 18 / 2 / 4 / 2 / 6 / 8 / 60

## Curriculum stages (widget_7 STAGES) — gen/code/reason/lctx/indic/stem

- Seed (warm start): 55/15/3/2/15/10
- General (broad base): 45/20/6/3/16/10
- Reasoning (code+logic): 25/28/18/5/14/10
- Long-context (stretch): 18/30/16/18/12/6
- Anneal (low-LR cool): 8/22/20/10/30/10
- **Reserved for cooldown:** Indic + reasoning Tier-A.
- **Difficulty ladder (B0→B5):** Nursery, Grade-school, High-school, Undergraduate, Graduate, Research/PhD.

## Training-stage budgets (widget_4)

Pretraining ~95% · Mid-training/Anneal ~2% · SFT <1% · Reasoning training <1% · Preference <1%.

## OPUS selector (widget_3)

Keep 40% default · English-heavy proxy vec [0.70,0.53,0.33] vs balanced [0.577³] ·
LANE_FLOOR **8%** reserved for Indic+agentic. V4: kept ~40%, ~6× effective tokens, few-% overhead.

## V4 anchors (lesson §2, §10)

Growth-stage rebalance: web 70→18% · code 13→35% · sci/math 7→39% · protected channel fixed 8%.
Instability: sudden Hindi-share jump × frozen embeddings → grad-norm spiked ~150×. Mitigation: warmup
every transition across several-billion-token bands.
