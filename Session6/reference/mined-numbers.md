# Session 6 — mined reference numbers (from the course widgets)

Extracted from `reference/widgets/*.html` (15 widgets) and `reference/session6-lesson.txt`.
Session 6 is a **coding assignment** (Training Data Execution System), not a written plan —
these are schema/default anchors for the implementation, not numbers to cite in prose.

## Worked example from lesson §2 (batch vocabulary)

8 GPUs × microbatch 2 × grad-accum 16 = **256 sequences/optimizer step**.
At seq_len 8,192: 256 × 8,192 = **2,097,152 token positions/step**.

## widget_2 batch_builder — defaults

gpus=8, micro=2, accum=16, seq=4096, ckpt_interval=500, toy_dataset=5B tokens.
Ranges: gpus 1-64, micro 1-8, accum 1-64, seq 512-32768 (step 512), ckpt 50-2000, dataset 1-50B.
global_batch = gpus×micro×accum; step_tokens = global_batch×seq; ckpt_tokens = step_tokens×ckpt_interval.

## widget_3 document_to_batch — 4 lane demo docs

web / code / indic / agent lanes, seq_len=24 for the packed-window demo. Agentic lane masks
`user` and `observation` tokens from loss when "loss mask" toggle is on — only plan/tool-call/
response tokens are loss-bearing (matches lesson §3).

## widget_4 padding_lab — 4 modes

right / left / crop / concat("fill with next doc"). Metrics: useful-position %, pad count,
cropped count, wasted-position % (rough = pad/ctx).

## widget_5 packing_simulator — 5 policies

pad-only / concat-and-chop / greedy / best-fit / structure-preserving. Toy doc lengths
generated as `8 + (i*17+13)%47` (range 8-54 tokens). Reports window utilization %, sequence
count, unused positions, boundary risk (none/medium/high/low by policy).

## widget_6 shard_manifest_builder — admission gate

Hard-required fields (block admission if missing): `tokenizer_hash`, `cleaning_pipeline_hash`,
`eval_overlap_status` (contam), license != unsafe. Soft fields: `dedup_status`, `pii_screen_status`,
`parent_manifest_ids`. Score = 78%×(passed-required-fraction) + 22%×license_score(safe=1/review=.5/unsafe=0),
capped at 64 if any hard requirement is blocked. Admit if score>86 → "Admitted to registry",
else "Held for review". Manifest JSON shape:
```
{shard_id, capability_lane, token_count, tokenizer_hash, content_hash,
 cleaning_pipeline_hash, dedup_status, pii_screen_status, eval_overlap_status,
 license_tier, parent_manifest_ids, admission}
```

## widget_7 mixture_timeline_compiler — toy supply + stages

Toy SUPPLY_T (widget's own demo numbers, NOT Session 5's real supply): General 95B, Code 38B,
Indic 18B, Reasoning 22B, Agentic 10B.
Stage bands (fraction of budget): **Foundation 0-55%**, **Skill build 55-85%**, **Anneal 85-100%**.
Profiles (General/Code/Indic/Reasoning/Agentic %): balanced 45/20/12/15/8, code-heavy
34/34/10/14/8, indic-heavy 38/18/22/14/8, anneal 36/21/14/19/10 (anneal profile = balanced
with General -8, Reasoning +5, Agentic +3 applied on top).
Feasibility check: `need = budget×share/100/(1-opus_reject_rate)`; warn if `supply < need`.

## widget_8 training_consumption_ledger — event shape

Ledger event (`batch_committed`): `ledger_offset, run_branch_id, global_step, checkpoint_id,
created_at, rank, microbatch_count, packed_sample_ids[], shard_ids[], token_span_ids[],
loss_mask_hash, position_policy, mixture_lane, curriculum_stage, opus_decision_id`.
`checkpoint_bound` event adds: `model_state, optimizer_state, dataloader_state
("ledger_offset_N"), rng_state`. Curriculum stage derived from step: <6 foundation,
<12 skill_build, else anneal (toy thresholds — real system derives from token position
against widget_7's stage bands).

## widget_9 checkpoint_comparison_lab

Confirms lesson §9's core claim numerically: comparison confidence without ledger =
`max(0, 100 - drift% - opus%)`; with-ledger replay = **100% stream identity** always.

## widget_10 opus_audit_board — decision taxonomy

4 buckets: accepted / rejected / deferred / protected.
Rejection reasons: `below_proxy_threshold, lane_quota_full, duplicate_update_direction,
stage_mismatch, eval_firewall_overlap, deferred_for_anneal`.
Protected-floor override fires when a candidate from the protected lane would otherwise be
rejected (except for firewall overlap, which always blocks regardless of protection).
Candidate record: `opus_decision_id, candidate_batch_id, model_age, proxy_version, lane,
stage, score, decision, rejection_reason, shard_ids[], effective_token_estimate`.

## widget_11 token_perplexity_heatmap — phase scaling

Loss multiplier by model phase: early ×1.35, mid ×0.95, late ×0.72, anneal ×0.62.
Lane adjustments: indic +0.55 (except in anneal phase), agentic tool-tokens +0.35,
ellipsis/"..." tokens +0.8, boundary tokens +0.2 (optional toggle). Non-loss-bearing tokens
get loss scaled to 28% (still shown, not zero, to make masking visible). Perplexity color
bands: <6 green, <14 teal, <28 amber, <60 red, else violet (calmer color = "shouldn't be this
hard anymore").

## widget_12 shard_learning_report_card — 5 example shards

`indic-tier-a-042` (Indic, 84M, opus 82, grad 71): strong late learner, preserve for anneal.
`code-repair-118` (Code, 51M, opus 74, grad 64): useful mid-training, diminishing anneal returns.
`agentic-browser-021` (Agentic, 19M, opus 68, grad 78): hard but productive, keep obs masked.
`web-clean-900` (Web, 420M, opus 48, grad 38): valuable early, weak late, don't spend anneal budget here.
`reason-proof-077` (Reasoning, 12M, opus 91, grad 86): looks too hard early, excellent once base model ready.
Usefulness score = `delta_loss×220×w + support×(1-w)` where
`support = (100-hot_ppl%)×.45 + grad×.25 + opus×.3`, w = user-set "loss delta weight" slider.
Badge: ≥72 useful / ≥48 review / else delay.

## widget_13 eval_firewall — 7 candidate shards + 4 gates

Gates (each independently toggleable): `never_train` flag, benchmark overlap >25%, canary
match, benchmark-derived content. Any tripped gate blocks. Example candidates span
overlap 0.01 (indic-news, clean) to 0.91 (mmlu-mirror-3, blocked, canary=true, never=true).
`gsm8k-rationale-blog` (overlap 0.32, derived=true, never=false) — shows a *derived* benchmark
explanation blog gets blocked on the derived-content gate even without a never_train flag.

## widget_14 crash_replay_fork — 3 modes

`ledger` (default): recovery replays historical sample stream from ledger offset exactly.
`random`: no ledger binding, next-batch stream shifts by a pseudo-random offset (+9) after
crash — same checkpoint, silently different data.
`fork`: intentional new branch, offset shifts by +17 and branch id changes run-a→run-b.
Checkpoint interval in the demo = every 200 steps (`ckpt = floor(step/200)`).

## widget_15 dataloader_throughput_lab — defaults + formula sketch

Defaults: seq=4096, global_batch=256 seq, packing=86%, opus_reject=18%, workers=18,
prefetch=6, shard=1024MB, bandwidth=1800MB/s, decompression_cost=24%.
`useful_after_pack = tokens_per_step × pack%`; `useful_after_opus = useful_after_pack ×
(1-reject%)`. `worker_factor = min(1.8, .35 + log2(workers+1)/4 + prefetch/30)`.
`shard_penalty`: <256MB → 0.72, >3072MB → 0.88, else 1.0 (small-shard and huge-shard both
penalized). Advice thresholds: pack<75% → "packing is the bottleneck"; reject>45% → "OPUS
discarding too much"; idle>20% → "GPU waiting, fix cache/prefetch/bandwidth"; shard<256MB →
"bundle into larger immutable objects".

## Assignment shape (§16, verbatim structure)

Full path required: `documents → tokenized shards → manifests → mixture schedule → packing →
batches → training → consumption ledger → learning ledger → checkpoint → crash → resume →
replay → audit`.

Deliverable: GitHub repo, one command (e.g. `python run_demo.py`) that regenerates
`submission_artifacts/{run.log, evidence.json, evidence.md, manifests/, ledgers/,
checkpoints/, performance.json}` with no manual steps. Evidence must be generated by the
implementation, not hardcoded — graders re-run the command and inspect code.

1,000 points: end-to-end execution 150, shards/manifests/tokenizer 100, packing/masks/batches
150, mixture/floors/OPUS 150, ledgers 150, checkpoint/crash/resume/replay/fork 150, eval
firewall 50, throughput 50, tests/evidence/docs 50.
