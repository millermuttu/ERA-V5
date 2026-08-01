## Why

A review of the shipped Session 6 system (`24e7f4f`) found that several of the
invariants the rubric grades most heavily were reporting PASS without actually
holding. The evidence bundle was honest about what it measured; the problem was
that what it measured was, in places, guaranteed to succeed.

- The crash drill halted exactly on a checkpoint boundary, so nothing was ever
  in flight and "no batch skipped or repeated" held trivially. A crash one batch
  later duplicated ledger offsets 10-13.
- Firewall-tripping shards scored 100 and came out `admitted`; nothing leaked
  only because `build_world` happened to iterate training shards alone.
- OPUS blended the lane's mixture weight into its proxy score, double-counting
  `pick_lane`'s sampling. Acceptance became impossible below a 1/3 lane share,
  so only `general` passed on merit and the realized stream collapsed to 86%
  general against a planned 45%.
- `token_span_ids` recorded the padded sample length rather than real document
  spans, and the packing-correctness evidence row was the tautology
  `0 <= pct <= 100`.
- `benchmark_overlap_pct`, `canary_match`, `dedup_status` and
  `pii_screen_status` were constants and default arguments. All 67 manifests
  asserted `dedup_status="passed"` with no dedup code anywhere in the repo.

The rubric's Step 3 inspects code specifically to confirm "the required
behaviour was not simulated or hardcoded", so these are scoring risks, not
cosmetic ones.

## What Changes

- **Crash recovery**: the drill crashes mid-interval and `JsonlLedger.truncate_after`
  rolls both ledgers back to the last durable checkpoint before resuming.
- **Firewall**: `decide_admission` blocks firewall-tripping manifests directly,
  and near-duplicates are blocked from admission.
- **OPUS**: the proxy score is independent of lane weight; the 25-attempt retry
  loop that re-rolled the candidate id until the hash cleared is removed; every
  final verdict (including rejected and deferred) is written to the audit
  ledger; `protected_floor_override` is recorded whenever the floor steers lane
  selection, not only when it rescues a candidate.
- **Packing/batches**: candidate batches draw `docs_per_batch=3` documents so
  segment-id attention blocking and per-document position resets are exercised
  by the committed stream; `token_span_ids` records real `shard:start:end`
  spans; replay reconstructs by slicing those spans out of the content-addressed
  shard rather than re-running the packer.
- **Measured cleaning**: Session 4's cleaning pipeline is vendored into
  `Session6/pipeline/cleaning.py` (stdlib-only) so eval overlap, canary hits,
  PII and near-duplicates are measured. The toy corpus plants real defects for
  those detectors to find, since it previously contained none. PII is masked
  before the vocabulary is built, closing a leak into `tokenizer.json`.
- **Reporting/evidence**: packing utilization is measured on the stream actually
  trained; the mixture row compares planned versus actual shares against the
  stage-weighted plan; the packing row re-verifies every packed sample; a new
  Data cleaning row covers the measured fields. Tamper tests assert each row
  flips to FAIL when its artifact is corrupted.
- **Artifacts**: `submission_artifacts/` is committed, as the assignment
  requires the generated log, evidence bundle, manifests, ledgers and
  checkpoints be present in the repository.

## Capabilities

### Modified Capabilities
- `data-execution-system`: tightens the admission, firewall, mixture, OPUS,
  packing, crash-recovery, throughput and evidence requirements so each is
  falsifiable rather than satisfiable by construction.

### New Capabilities
(none - this hardens the existing capability)

## Impact

- `Session6/pipeline/`: new `cleaning.py`; changes to `batch.py`, `checkpoint.py`,
  `corpus.py`, `crash_resume.py`, `evidence.py`, `ledger.py`, `manifest.py`,
  `mixture.py`, `opus.py`, `packing.py`, `replay.py`, `shards.py`, `train_loop.py`.
- `Session6/run_demo.py`: cleaning + checkpoint log events, fork ordered before
  audit, packed-batch report, branch-scoped checkpoint lookup.
- Tests: 45 -> 74, including a new `test_cleaning.py`.
- New artifact `submission_artifacts/packed_batches.json` (the rubric's
  "packed-batch report"); `submission_artifacts/` no longer gitignored.
- Dependencies: still stdlib only. Session 4's MinHash dedup was deliberately
  not carried over, to avoid taking a numpy dependency for 67 documents.
- Reported numbers move because they are now measured on the real stream:
  packing utilization 71.74% -> 41.2%, useful tokens/sec ~15.2k -> ~21.9k.
