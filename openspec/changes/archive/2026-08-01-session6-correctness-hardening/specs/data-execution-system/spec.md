## Purpose

Defines the observable behavior of the toy-scale training-data execution pipeline: the guarantees it must uphold from raw documents through packed batches, ledgers, checkpoints, crash/resume, replay, and audit, so that a single command produces a verifiable, non-hardcoded evidence bundle.

## MODIFIED Requirements

### Requirement: Shard Manifest Admission Gate
The system SHALL admit a shard only if all hard-required fields (`tokenizer_hash`, `cleaning_pipeline_hash`, `eval_overlap_status`) are present, its `license_tier` is not `unsafe`, its `eval_overlap_status` is `clear`, it is not a near-duplicate of an already-seen shard, and its computed admission score exceeds the admission threshold. Admission SHALL NOT depend on any downstream consumer declining to select the shard.

#### Scenario: Shard missing a hard-required field is not admitted
- **WHEN** a shard manifest is missing `cleaning_pipeline_hash` or has `license_tier: unsafe`
- **THEN** the manifest's `admission` is `blocked` or `held_for_review`, never `admitted`

#### Scenario: Complete high-scoring shard is admitted
- **WHEN** a shard manifest has all hard-required fields present, a `clear` eval-overlap status, and an admission score above the threshold
- **THEN** the manifest's `admission` is `admitted`

#### Scenario: Firewall-tripping shard is blocked by the admission gate itself
- **WHEN** a shard's `eval_overlap_status` is not `clear`
- **THEN** its admission score is capped at the hard-block cap and its `admission` is `blocked`, independently of whether any training loop would have selected it

#### Scenario: Near-duplicate shard is blocked
- **WHEN** a shard's content is a near-duplicate of an earlier admitted shard
- **THEN** its `dedup_status` records `near_duplicate_of:<shard_id>` and its `admission` is `blocked`

### Requirement: Evaluation and Validation Firewall
The system SHALL block any shard that is flagged `never_train`, declared benchmark-derived, measured as exceeding the benchmark-overlap threshold, or measured as containing a canary string, from ever appearing in a loss-bearing training batch, regardless of its mixture lane's protected status. Benchmark overlap and canary presence SHALL be measured from the document text, not read from a declared field.

#### Scenario: Firewall-flagged shard is blocked at admission
- **WHEN** a candidate shard trips any of the firewall's four gates
- **THEN** the shard's manifest `admission` is `blocked`, the shard never appears in any `batch_committed` event, and `run.log` contains `[PASS] eval_shard_blocked`

#### Scenario: Protected lane does not override the firewall
- **WHEN** a firewall-flagged shard belongs to a lane currently under its protected floor
- **THEN** the shard is still rejected by the firewall and is never granted a protected-floor override

#### Scenario: Overlap is measured from text, not declared
- **WHEN** a training document contains a verbatim span copied from an evaluation document
- **THEN** its `benchmark_overlap_pct` is computed from n-gram fingerprint overlap against the evaluation set, and exceeds the configured threshold without any per-document overlap value having been declared in the corpus

### Requirement: Mixture Schedule with Protected Floors
The system SHALL compile a mixture schedule with named curriculum stages (Foundation, Skill-build, Anneal) and per-lane target shares, SHALL never let a lane's realized share fall below its configured protected floor over the course of a run, and SHALL keep each lane's realized token share within a bounded drift of the stage-weighted planned share so that the stream cannot collapse onto a single lane.

#### Scenario: Lane share stays at or above its floor
- **WHEN** a full training stream is materialized against a mixture schedule with protected floors
- **THEN** every lane's cumulative realized token share is greater than or equal to its configured floor at the end of the run

#### Scenario: Realized shares track the planned mixture
- **WHEN** a full training stream is materialized
- **THEN** every lane's realized token share is within the configured maximum drift of its stage-weighted planned share, where the planned share is each stage profile weighted by that stage band's width

#### Scenario: No lane is structurally excluded
- **WHEN** candidate batches are scored in any curriculum stage
- **THEN** every lane defined in that stage's profile is capable of being `accepted` on its proxy score alone, without relying on the protected-floor override to enter the stream

### Requirement: OPUS Admission Decisions
The system SHALL classify every candidate batch into exactly one of `accepted`, `rejected`, `deferred`, or `protected`, SHALL record a `rejection_reason` for every `rejected` decision, SHALL produce the same decision for the same candidate id across repeated runs, SHALL score each candidate batch exactly once, SHALL compute its proxy score independently of the candidate's mixture lane weight, and SHALL persist every final verdict — including `rejected` and `deferred` candidates that never become batches — to the decision audit ledger.

#### Scenario: Deterministic decision for a fixed candidate
- **WHEN** the same candidate batch id is scored twice under identical mixture/stage state
- **THEN** both scoring runs produce the same decision and score

#### Scenario: Rejected decision always carries a reason
- **WHEN** a candidate batch is decided `rejected`
- **THEN** its record's `rejection_reason` is a non-null value from the defined reason enum

#### Scenario: Proxy score does not re-apply the mixture
- **WHEN** the same candidate id is scored for different lanes and stages
- **THEN** the proxy score is identical, because lane selection has already applied the mixture upstream

#### Scenario: Audit ledger contains rejected and deferred candidates
- **WHEN** a run completes
- **THEN** the decision audit ledger contains one record per candidate batch scored, including those decided `rejected` and `deferred`, not only those that became committed batches

#### Scenario: Protected-floor override is recorded whenever the floor steers selection
- **WHEN** the protected floor forces a lane to be selected
- **THEN** the resulting decision records `protected_floor_override: true`, whether the proxy score would have accepted the candidate anyway or the floor had to rescue it from rejection

### Requirement: Correct Packing, Masks, and Position IDs
The system SHALL pack tokens into batches using a lane-appropriate policy, SHALL compute a `loss_mask` that excludes padding and non-loss-bearing positions, SHALL represent cross-document attention blocking within a packed sample as per-token segment ids such that a token may attend only to earlier tokens carrying the same segment id, SHALL reset `position_ids` to zero at each document boundary, and SHALL produce packed samples containing more than one document during a normal run so these guarantees are exercised by the committed stream rather than only in isolation.

#### Scenario: Structure-preserving policy never mixes documents
- **WHEN** a bin is packed using the `structure_preserving` policy
- **THEN** the bin contains tokens from exactly one source document

#### Scenario: Loss mask excludes padding
- **WHEN** a packed sample contains padding tokens
- **THEN** the `loss_mask` value at every padding position is zero

#### Scenario: Position ids reset at document boundaries
- **WHEN** a packed sample contains more than one document
- **THEN** the `position_ids` sequence restarts from zero at the start of each document segment

#### Scenario: Attention is blocked across document boundaries
- **WHEN** two tokens in a packed sample belong to different document segments
- **THEN** neither token may attend to the other, and a token may attend only to earlier positions sharing its segment id

#### Scenario: The committed stream contains multi-document samples
- **WHEN** a full run is materialized
- **THEN** at least one committed batch contains more than one source document

### Requirement: Checkpoint, Crash, and Resume Without Skip or Repeat
The system SHALL be able to save a checkpoint bound to a ledger offset, deliberately halt execution partway through a run at a point that is not a checkpoint boundary, discard ledger entries written after the last durable checkpoint, and resume such that the concatenation of batches processed before the crash and after the resume exactly equals the original uninterrupted batch sequence, with no batch skipped or repeated.

#### Scenario: Crash lands between checkpoints
- **WHEN** the crash point is chosen for the drill
- **THEN** it falls strictly between two checkpoint boundaries, so that batches committed after the last durable checkpoint are in flight and must be reconciled

#### Scenario: In-flight batches are rolled back before resume
- **WHEN** the run crashes after committing batches beyond the last durable checkpoint
- **THEN** those consumption- and learning-ledger entries are discarded on resume, and each of those batches is subsequently committed exactly once

#### Scenario: Resume continues at the expected next batch
- **WHEN** the run is resumed from the last durable checkpoint
- **THEN** the first batch processed after resume has the exact batch id, token spans, and loss-mask hash expected to follow that checkpoint in the original plan, and `run.log` contains `[PASS] resume_next_batch_matched`

#### Scenario: Checkpoint lookup is scoped to a run branch
- **WHEN** more than one run branch has written checkpoints into the same checkpoint directory
- **THEN** requesting the latest checkpoint for a given `run_branch_id` returns that branch's checkpoint, never another branch's

### Requirement: Historical Replay with Hash Verification
The system SHALL be able to replay a previously committed interval in ledger mode and reconstruct the same batch ids, token spans, and loss-mask hashes as the original run. Replay SHALL reconstruct each packed sample from the token spans recorded in the ledger and the content-addressed shards, and the recorded spans SHALL describe real document extents rather than the padded sample length.

#### Scenario: Ledger-mode replay reproduces identical hashes
- **WHEN** an earlier committed interval is replayed in ledger mode
- **THEN** the reconstructed batch ids, token spans, and loss-mask hashes are identical to those recorded during the original run, and `run.log` contains `[PASS] replay_hash_matched`

#### Scenario: Token spans describe real document extents
- **WHEN** a batch is committed
- **THEN** each recorded `token_span_ids` entry describes that document's actual extent within its shard's token stream, and is not the padded sample length

#### Scenario: A tampered span does not silently reconstruct
- **WHEN** a recorded token span is altered and then replayed
- **THEN** the reconstructed sample's loss-mask hash differs from the recorded hash, rather than being re-derived from the packer

### Requirement: Packing Efficiency and Throughput Reporting
The system SHALL compute packing utilization from the stream that was actually committed during the run, SHALL compute useful loss-bearing tokens-per-second from that utilization together with the real OPUS accept/reject counts, and SHALL store the inputs used so the reported numbers can be independently recomputed. Reported packing utilization SHALL be reconstructable from the token spans recorded in the consumption ledger.

#### Scenario: Performance report is reconstructable from its own inputs
- **WHEN** `performance.json` is generated at the end of a run
- **THEN** recomputing its formulas from the `inputs` it records reproduces the same output values

#### Scenario: Reported utilization matches the committed stream
- **WHEN** packing utilization is recomputed from the token spans and sample capacities recorded for every committed batch
- **THEN** the recomputed value equals the utilization reported in `performance.json`

### Requirement: Generated, Non-Hardcoded Evidence Bundle
The system SHALL produce `evidence.json` and `evidence.md` by scanning the artifacts actually generated during the run (manifests, ledgers, checkpoints, packed-batch report, performance report), SHALL cite a real artifact path as evidence for every requirement row, SHALL NOT contain any hardcoded pass/fail result that does not derive from the generated artifacts, and SHALL derive each row from a check that can fail — no row may be satisfied by construction.

#### Scenario: Every evidence row cites an existing artifact
- **WHEN** `evidence.json` is generated after a full run
- **THEN** every row's cited evidence path exists under the run's `submission_artifacts/` directory

#### Scenario: Evidence reflects an induced failure
- **WHEN** a required invariant is deliberately broken before running the demo
- **THEN** the corresponding evidence row reports `FAIL`, not a hardcoded `PASS`

#### Scenario: Packing evidence re-verifies the packed samples
- **WHEN** the packing-correctness row is computed
- **THEN** it re-derives each packed sample's loss-mask hash, confirms no padding position is loss-bearing, confirms position ids reset per document, confirms segment ids are non-decreasing, and confirms recorded span lengths match the token runs they describe

#### Scenario: Corrupting a packed sample fails the packing row
- **WHEN** a recorded loss mask, position-id reset, or reported utilization is altered in the packed-batch report
- **THEN** the packing-correctness row reports `FAIL`

## ADDED Requirements

### Requirement: Measured Data Cleaning
The system SHALL determine each shard's benchmark overlap, canary presence, PII content, and near-duplicate status by running detection over the document text, and SHALL record the results on the shard manifest. These fields SHALL NOT be constants, default arguments, or values declared alongside the source documents. Provenance attributes that are not derivable from text (`never_train`, `benchmark_derived`) MAY remain declared.

#### Scenario: Cleaning statuses vary across the corpus
- **WHEN** manifests are generated for a corpus containing both clean and defective documents
- **THEN** `dedup_status` and `pii_screen_status` each take more than one distinct value across the manifest set

#### Scenario: PII is detected and counted
- **WHEN** a document contains structured identifiers such as an email address or phone number
- **THEN** its manifest records `pii_screen_status: masked` with non-zero per-category counts, and the identifiers are replaced with placeholders before tokenization

#### Scenario: Near-duplicates are detected by content similarity
- **WHEN** a document is a near-copy of an earlier document
- **THEN** it is identified by shingle-based Jaccard similarity above the configured threshold, and the later document is the one flagged

#### Scenario: The corpus contains defects for the detectors to find
- **WHEN** the toy corpus is generated
- **THEN** it contains at least one document exceeding the eval-overlap threshold, at least one canary hit, at least one document with PII, and at least one near-duplicate, so that each detector's result is a measurement rather than a vacuous zero

### Requirement: PII Masked Before Vocabulary Construction
The system SHALL mask structured personal identifiers before the tokenizer vocabulary is built, so that no identifier appears in the published tokenizer artifact.

#### Scenario: Identifiers never enter the vocabulary
- **WHEN** the corpus contains documents with email addresses, phone numbers, IP addresses, or URLs
- **THEN** none of those identifier strings appear in the frozen tokenizer's vocabulary or in `manifests/tokenizer.json`
