## 1. Crash recovery

- [x] 1.1 Add `JsonlLedger.truncate_after` to discard events past a ledger offset
- [x] 1.2 Add `ledger_offset` to learning-ledger events so they can be rolled back too
- [x] 1.3 Crash mid-interval (`default_crash_offset`) instead of on a checkpoint boundary
- [x] 1.4 Roll both ledgers back to the last durable checkpoint before resuming
- [x] 1.5 Scope `latest_checkpoint`/`first_checkpoint` to a `run_branch_id`
- [x] 1.6 Test a mid-interval crash duplicates nothing; test the crash point is not a boundary

## 2. Firewall and admission

- [x] 2.1 Block firewall-tripping manifests in `decide_admission`; cap their score
- [x] 2.2 Record `eval_firewall_reasons` on the manifest
- [x] 2.3 Block near-duplicate shards from admission
- [x] 2.4 Test every firewall-flagged eval shard is blocked, not merely unselected

## 3. OPUS

- [x] 3.1 Make the proxy score independent of lane weight
- [x] 3.2 Remove the 25-attempt candidate-id retry loop
- [x] 3.3 Restrict the anneal deferral to the foundation stage
- [x] 3.4 Persist every final verdict, including rejected and deferred, to the audit ledger
- [x] 3.5 Record `protected_floor_override` whenever the floor steers lane selection
- [x] 3.6 Test no lane is structurally unacceptable in any stage
- [x] 3.7 Test realized lane shares track the stage-weighted plan

## 4. Packing, masks, spans

- [x] 4.1 Draw `docs_per_batch` documents per candidate batch
- [x] 4.2 Handle a candidate batch yielding multiple bins
- [x] 4.3 Carry `start`/`end` spans on every packed document
- [x] 4.4 Emit `token_span_ids` from `build_masked_sample`; record them in the ledger
- [x] 4.5 Size samples by `bin_capacity` so long-context bins are not truncated
- [x] 4.6 Reconstruct replay from recorded spans instead of re-running the packer
- [x] 4.7 Emit one learning rollup per document in a bin
- [x] 4.8 Test spans are real extents; test a tampered span changes the hash

## 5. Measured cleaning

- [x] 5.1 Vendor Session 4's decontam, structured-PII and shingle code into `cleaning.py`
- [x] 5.2 Add `overlap_pct`; use exact Jaccard rather than MinHash at toy scale
- [x] 5.3 Plant contamination, canary, PII and near-duplicates in the corpus
- [x] 5.4 Measure `benchmark_overlap_pct` and `canary_match` in `build_shard`
- [x] 5.5 Derive `dedup_status` and `pii_screen_status` from what was found
- [x] 5.6 Mask PII before building the tokenizer vocabulary
- [x] 5.7 Record measured values on the manifest for auditability
- [x] 5.8 Test the corpus still contains defects and they still block admission

## 6. Reporting and evidence

- [x] 6.1 Measure packing utilization on the committed stream
- [x] 6.2 Write `packed_batches.json` as the rubric's packed-batch report
- [x] 6.3 Replace the `0 <= pct <= 100` packing row with full re-verification
- [x] 6.4 Add `planned_lane_shares`; compare planned versus actual in the mixture row
- [x] 6.5 Require rejections, deferrals and floor overrides in the OPUS row
- [x] 6.6 Add a Data cleaning evidence row
- [x] 6.7 Add tamper tests asserting each row flips to FAIL

## 7. Log, artifacts, docs

- [x] 7.1 Add the missing `checkpoint saved` event; split its `[PASS]` from resume's
- [x] 7.2 Order `branch forked` before `audit completed` per the assignment sequence
- [x] 7.3 Add a `data cleaned` event
- [x] 7.4 Commit `submission_artifacts/`; negate the root `*.log` rule for `run.log`
- [x] 7.5 Update `Session6/README.md` with measured results and the design decisions
- [x] 7.6 Verify the full assignment checklist and confirm byte-identical reruns
