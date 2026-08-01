## 1. Scaffolding

- [ ] 1.1 Create `Session6/pipeline/` package with `__init__.py` and `Session6/pipeline/tests/` with `conftest.py` (mirrors `Session4/pipeline/conftest.py`'s `sys.path` setup)
- [ ] 1.2 Add `Session6/.gitignore` entry for `submission_artifacts/`

## 2. Corpus and Tokenizer

- [ ] 2.1 Implement `corpus.py`: deterministic seeded generator producing documents across 5 training lanes (general/code/indic/reasoning/agentic) plus eval/benchmark documents modeled on the firewall's 7 candidate shapes (varying overlap %, canary match, derived-content flags)
- [ ] 2.2 Implement `tokenizer.py`: whitespace/punctuation vocab built from the generated corpus + special tokens, `tokenizer_hash` via `sha256`, frozen and written to `manifests/tokenizer.json`
- [ ] 2.3 Write `tests/test_tokenizer.py`: same text -> same ids; vocab change -> different `tokenizer_hash`

## 3. Shards and Manifests

- [ ] 3.1 Implement `shards.py`: content-addressed immutable shard construction (`shard_id`/`content_hash` via `sha256` of tokenized content)
- [ ] 3.2 Implement `manifest.py`: shard manifest schema (per design.md) and admission-gate scoring (hard-required fields, cap-at-64 on hard-block, admit threshold > 86)
- [ ] 3.3 Write `tests/test_manifest.py`: admission score formula, hard-block cap, threshold behavior; re-tokenizing identical docs is idempotent, changing one doc changes the hash

## 4. Eval/Validation Firewall

- [ ] 4.1 Implement `firewall.py`: single canonical `check(shard)` function covering the 4 gates (`never_train`, benchmark-overlap > 25%, canary match, benchmark-derived content)
- [ ] 4.2 Write `tests/test_firewall.py`: each gate independently blocks; a clean shard passes; a derived-content-without-never_train shard is still blocked

## 5. Mixture Schedule

- [ ] 5.1 Implement `mixture.py`: stage bands (Foundation 0-55%, Skill-build 55-85%, Anneal 85-100%) keyed to cumulative tokens consumed / toy budget, lane profiles reusing Session5's percentages, protected-floor enforcement, feasibility check against real measured shard supply
- [ ] 5.2 Write `tests/test_mixture.py`: protected floor holds over many draws; stage bands partition correctly

## 6. OPUS Admission

- [ ] 6.1 Implement `opus.py`: hash-seeded deterministic scoring, decision taxonomy (accepted/rejected/deferred/protected), rejection-reason enum, firewall-flagged candidates always rejected regardless of protected-lane status
- [ ] 6.2 Write `tests/test_opus.py`: firewall override never happens; decision/score reproducible across repeated calls for a fixed candidate id

## 7. Packing and Batches

- [ ] 7.1 Implement `packing.py`: pad_only, concat_and_chop, greedy, best_fit, structure_preserving, long_context (as a tagged pad_only variant) policies, with a lane -> policy default map
- [ ] 7.2 Implement `batch.py`: `loss_mask`/`attention_mask`/`position_ids` construction (position reset per document, attention blocked across document boundaries, loss mask excludes padding and non-loss-bearing roles), microbatch/global batch assembly, `loss_mask_hash`
- [ ] 7.3 Write `tests/test_packing.py`: per-policy utilization/bin-count recomputation, concat_and_chop token accounting, structure_preserving never mixes two docs in one bin
- [ ] 7.4 Write `tests/test_masks.py`: loss_mask excludes pad + masked roles; attention blocked across segments; position_ids reset per doc

## 8. Ledgers

- [ ] 8.1 Implement `ledger.py`: append-only JSONL consumption ledger (`batch_committed`, `checkpoint_bound` events) and learning ledger (per-shard/step loss rollups)
- [ ] 8.2 Write `tests/test_ledger.py`: append-only, strictly increasing `ledger_offset`, `checkpoint_bound` references a valid offset

## 9. Checkpointing

- [ ] 9.1 Implement `checkpoint.py`: checkpoint record save/load tied to `ledger_offset`
- [ ] 9.2 Write `tests/test_checkpoint_resume.py` scaffolding for the record shape (full crash/resume behavior test lives in section 11)

## 10. Training Loop

- [ ] 10.1 Implement `train_loop.py`: materialize the full mixture -> OPUS -> packing plan as an ordered list of batch records indexed by `ledger_offset`; deterministic toy loss computation; periodic learning-ledger rollups; checkpoint saves every `ckpt_interval` steps

## 11. Crash, Resume, Replay, Fork, Audit

- [ ] 11.1 Implement `crash_resume.py`: deliberate mid-stream halt, resume from last checkpoint's `ledger_offset`, assert resumed batch id matches `plan[ledger_offset]` exactly (no skip, no repeat)
- [ ] 11.2 Write `tests/test_checkpoint_resume.py`: concatenation of pre-crash + post-resume batch ids equals the original uninterrupted plan
- [ ] 11.3 Implement `replay.py`: ledger mode (exact reconstruction + hash equality assertion), random mode (+9 offset, documented negative example), fork mode (+17 offset, new `run_branch_id`)
- [ ] 11.4 Write `tests/test_replay.py`: ledger-mode hash equality; random/fork offset-shift formulas
- [ ] 11.5 Implement `audit.py`: reconstruct shards/OPUS decisions behind a given checkpoint or token range

## 12. Throughput and Evidence

- [ ] 12.1 Implement `throughput.py`: `tokens_per_step`, `useful_after_pack`, `useful_after_opus`, `worker_factor`, `shard_penalty`, `useful_tokens_per_sec`, `packing_utilization_pct` computed from real measured run stats, with `inputs` stored in `performance.json`
- [ ] 12.2 Write `tests/test_throughput.py`: recomputing formulas from stored `inputs` reproduces stored outputs
- [ ] 12.3 Implement `evidence.py`: scan generated `manifests/`, `ledgers/`, `checkpoints/`, `performance.json` to build `evidence.json`/`evidence.md`, citing real artifact paths, no hardcoded results
- [ ] 12.4 Write `tests/test_evidence.py`: every `evidence_path` cited in `evidence.json` exists under `submission_artifacts/`

## 13. Orchestration and Documentation

- [ ] 13.1 Implement `run_demo.py`: sequential orchestration (corpus -> tokenizer -> shards/manifests -> firewall -> mixture compile -> train loop with ledger writes and checkpoints -> deliberate crash -> resume+verify -> replay+verify -> fork -> audit -> throughput -> evidence -> exit code), writing `[PASS] <check_name>` lines to `run.log` for each verified invariant
- [ ] 13.2 Run `python run_demo.py` end-to-end; iterate until `submission_artifacts/` is fully generated and all core `[PASS]` checks succeed
- [ ] 13.3 Run `pytest Session6/pipeline/tests/ -q`; fix any failures
- [ ] 13.4 Write `Session6/README.md`: architecture overview, key design decisions (link back to design.md), exact reproduce commands (`python run_demo.py`, `pytest ...`), and a results table drawn from the real generated `performance.json`/`evidence.md`
