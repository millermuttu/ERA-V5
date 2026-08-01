## Purpose

Defines the observable behavior of the toy-scale training-data execution pipeline: the guarantees it must uphold from raw documents through packed batches, ledgers, checkpoints, crash/resume, replay, and audit, so that a single command produces a verifiable, non-hardcoded evidence bundle.

## ADDED Requirements

### Requirement: Immutable Content-Addressed Shards
The system SHALL derive each shard's identifier from a hash of its tokenized content, and re-tokenizing identical source documents SHALL always yield the same shard id and content hash.

#### Scenario: Identical documents produce identical shard id
- **WHEN** the same set of source documents is tokenized twice with the same frozen tokenizer
- **THEN** both runs produce shards with identical `shard_id` and `content_hash` values

#### Scenario: Changed document changes the shard hash
- **WHEN** a single source document in a shard is modified before tokenization
- **THEN** the resulting `content_hash` and `shard_id` differ from the original shard's

### Requirement: Frozen Tokenizer Hash Verification
The system SHALL record a `tokenizer_hash` derived from the frozen vocabulary, and every shard manifest SHALL be rejected if its tokenizer hash does not match the frozen tokenizer's hash.

#### Scenario: Tokenizer hash matches across the run
- **WHEN** shards are produced using the frozen tokenizer
- **THEN** every shard manifest's `tokenizer_hash` equals the tokenizer's recorded hash, and `run.log` contains `[PASS] tokenizer_hash_verified`

### Requirement: Shard Manifest Admission Gate
The system SHALL admit a shard only if all hard-required fields (`tokenizer_hash`, `cleaning_pipeline_hash`, `eval_overlap_status`, and a `license_tier` other than `unsafe`) are present and its computed admission score exceeds the admission threshold.

#### Scenario: Shard missing a hard-required field is not admitted
- **WHEN** a shard manifest is missing `cleaning_pipeline_hash` or has `license_tier: unsafe`
- **THEN** the manifest's `admission` is `blocked` or `held_for_review`, never `admitted`

#### Scenario: Complete high-scoring shard is admitted
- **WHEN** a shard manifest has all hard-required fields present and an admission score above the threshold
- **THEN** the manifest's `admission` is `admitted`

### Requirement: Evaluation and Validation Firewall
The system SHALL block any shard flagged as `never_train`, exceeding the benchmark-overlap threshold, matching a canary string, or derived from benchmark content, from ever appearing in a loss-bearing training batch, regardless of its mixture lane's protected status.

#### Scenario: Firewall-flagged shard is blocked at admission
- **WHEN** a candidate shard trips any of the firewall's four gates
- **THEN** the shard is excluded from the training stream and `run.log` contains `[PASS] eval_shard_blocked`

#### Scenario: Protected lane does not override the firewall
- **WHEN** a firewall-flagged shard belongs to a lane currently under its protected floor
- **THEN** the shard is still rejected by the firewall and is never granted a protected-floor override

### Requirement: Mixture Schedule with Protected Floors
The system SHALL compile a mixture schedule with named curriculum stages (Foundation, Skill-build, Anneal) and per-lane target shares, and SHALL never let a lane's realized share fall below its configured protected floor over the course of a run.

#### Scenario: Lane share stays at or above its floor
- **WHEN** a full training stream is materialized against a mixture schedule with protected floors
- **THEN** every lane's cumulative realized token share is greater than or equal to its configured floor at the end of the run

### Requirement: OPUS Admission Decisions
The system SHALL classify every candidate batch into exactly one of `accepted`, `rejected`, `deferred`, or `protected`, SHALL record a `rejection_reason` for every `rejected` decision, and SHALL produce the same decision for the same candidate id across repeated runs.

#### Scenario: Deterministic decision for a fixed candidate
- **WHEN** the same candidate batch id is scored twice under identical mixture/stage state
- **THEN** both scoring runs produce the same decision and score

#### Scenario: Rejected decision always carries a reason
- **WHEN** a candidate batch is decided `rejected`
- **THEN** its record's `rejection_reason` is a non-null value from the defined reason enum

### Requirement: Correct Packing, Masks, and Position IDs
The system SHALL pack tokens into batches using a lane-appropriate policy, SHALL compute a `loss_mask` that excludes padding and non-loss-bearing positions, SHALL compute an `attention_mask` that blocks attention across document boundaries within a packed sample, and SHALL reset `position_ids` to zero at each document boundary.

#### Scenario: Structure-preserving policy never mixes documents
- **WHEN** a bin is packed using the `structure_preserving` policy
- **THEN** the bin contains tokens from exactly one source document

#### Scenario: Loss mask excludes padding
- **WHEN** a packed sample contains padding tokens
- **THEN** the `loss_mask` value at every padding position is zero

#### Scenario: Position ids reset at document boundaries
- **WHEN** a packed sample contains more than one document
- **THEN** the `position_ids` sequence restarts from zero at the start of each document segment

### Requirement: Consumption and Learning Ledgers
The system SHALL append a `batch_committed` consumption-ledger event for every committed batch with a strictly increasing `ledger_offset`, SHALL append a `checkpoint_bound` event whenever a checkpoint is saved that references a valid `ledger_offset`, and SHALL record learning-ledger rollups linking loss statistics back to source shard ids.

#### Scenario: Ledger offsets strictly increase
- **WHEN** multiple batches are committed during a run
- **THEN** each successive `batch_committed` event's `ledger_offset` is strictly greater than the previous one

#### Scenario: Checkpoint binds to a valid ledger offset
- **WHEN** a checkpoint is saved
- **THEN** its `checkpoint_bound` event's `ledger_offset` corresponds to an actual committed batch in the consumption ledger

### Requirement: Checkpoint, Crash, and Resume Without Skip or Repeat
The system SHALL be able to save a checkpoint bound to a ledger offset, deliberately halt execution partway through a run, and resume from the last checkpoint such that the concatenation of batches processed before the crash and after the resume exactly equals the original uninterrupted batch sequence, with no batch skipped or repeated.

#### Scenario: Resume continues at the expected next batch
- **WHEN** the run crashes after committing batch N and is resumed from the checkpoint bound at batch N
- **THEN** the first batch processed after resume has the exact batch id expected to follow batch N in the original plan, and `run.log` contains `[PASS] resume_next_batch_matched`

### Requirement: Historical Replay with Hash Verification
The system SHALL be able to replay a previously committed interval in ledger mode and reconstruct the same batch ids, token spans, and loss-mask hashes as the original run.

#### Scenario: Ledger-mode replay reproduces identical hashes
- **WHEN** an earlier committed interval is replayed in ledger mode
- **THEN** the reconstructed batch ids, token spans, and loss-mask hashes are identical to those recorded during the original run, and `run.log` contains `[PASS] replay_hash_matched`

### Requirement: Fork From an Earlier Checkpoint
The system SHALL support forking a new run branch from an earlier checkpoint, continuing execution under a distinct `run_branch_id` without altering the original branch's recorded ledger history.

#### Scenario: Fork creates a distinct branch without mutating the original
- **WHEN** a fork is created from a checkpoint recorded under `run-a`
- **THEN** subsequent batches are recorded under a new `run_branch_id` (e.g. `run-b`) and `run-a`'s previously committed ledger events are unchanged

### Requirement: Packing Efficiency and Throughput Reporting
The system SHALL compute packing utilization and useful loss-bearing tokens-per-second from the actual measured run (real packing fill rates and real OPUS accept/reject counts), and SHALL store the inputs used so the reported numbers can be independently recomputed.

#### Scenario: Performance report is reconstructable from its own inputs
- **WHEN** `performance.json` is generated at the end of a run
- **THEN** recomputing its formulas from the `inputs` it records reproduces the same output values

### Requirement: Generated, Non-Hardcoded Evidence Bundle
The system SHALL produce `evidence.json` and `evidence.md` by scanning the artifacts actually generated during the run (manifests, ledgers, checkpoints, performance report), SHALL cite a real artifact path as evidence for every requirement row, and SHALL NOT contain any hardcoded pass/fail result that does not derive from the generated artifacts.

#### Scenario: Every evidence row cites an existing artifact
- **WHEN** `evidence.json` is generated after a full run
- **THEN** every row's cited evidence path exists under the run's `submission_artifacts/` directory

#### Scenario: Evidence reflects an induced failure
- **WHEN** a required invariant is deliberately broken before running the demo
- **THEN** the corresponding evidence row reports `FAIL`, not a hardcoded `PASS`
