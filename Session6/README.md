# Session 6 — Training Data Execution System (ERA V5)

A small but complete implementation of the full training-data path:
`documents -> tokenized shards -> manifests -> mixture schedule -> packing ->
batches -> training -> consumption ledger -> learning ledger -> checkpoint ->
crash -> resume -> replay -> fork -> audit`, at toy scale, proving the system
is correct, reproducible, and auditable rather than proving it scales.

**Deliverable:** `python run_demo.py` — regenerates `submission_artifacts/`
from scratch on every run. The copy committed to this repo is the output of
a real run, not a hand-written sample; the command deletes and rebuilds the
whole directory, so re-running it is the way to verify every number below.

## Architecture

18 focused, independently-tested modules under `pipeline/`, mirroring
Session 4's pipeline/tests layout:

| Module | Responsibility |
|---|---|
| `corpus.py` | Deterministic seeded toy corpus: 5 training lanes (general/code/indic/reasoning/agentic) + 7 eval/benchmark candidates, with real contamination/PII/near-duplicates planted for the cleaning pass to find |
| `cleaning.py` | Decontamination, PII screening and near-duplicate detection, vendored from Session 4's submitted cleaning pipeline (stdlib only) |
| `tokenizer.py` | Frozen whitespace/punctuation tokenizer with a `sha256` vocab hash (stdlib only — a real BPE isn't needed at this scale) |
| `shards.py` | Content-addressed, immutable shards (`shard_id`/`content_hash` from tokenized content) |
| `manifest.py` | Shard manifest schema + admission-gate scoring (hard-required fields, cap-at-64 on hard-block, admit threshold > 86) |
| `firewall.py` | Single canonical 4-gate eval/validation firewall, called at admission *and* batch assembly |
| `mixture.py` | Curriculum stage bands (Foundation/Skill-build/Anneal), lane profiles, protected-floor enforcement |
| `opus.py` | Deterministic (hash-seeded) admission scoring: accepted/rejected/deferred/protected |
| `packing.py` | 6 packing policies (pad_only, concat_and_chop, greedy, best_fit, structure_preserving, long_context); every packed document carries its `start`/`end` span into the shard's token stream |
| `batch.py` | Loss mask / segment-id attention blocking / position-id reset per document, plus the `token_span_ids` replay reconstructs from |
| `ledger.py` | Append-only JSONL consumption + learning ledgers |
| `checkpoint.py` | Checkpoint records tied to `ledger_offset`, tracked via an on-disk index |
| `train_loop.py` | Materializes the full mixture→OPUS→packing plan; runs the toy loss loop against it |
| `crash_resume.py` | Deliberate crash + resume, verified against the on-disk ledger |
| `replay.py` | Ledger / random / fork replay modes |
| `audit.py` | Reconstructs shards/OPUS decisions behind a checkpoint |
| `throughput.py` | Packing/throughput formulas, with `inputs` stored for independent recomputation |
| `evidence.py` | Scans the generated artifacts to build `evidence.json`/`evidence.md` — never a hardcoded result |

Full design rationale (why hash-seeded scoring instead of `random`, why
`long_context`/`structure_preserving` share `pad_only`'s implementation, why
checkpoint/resume/replay are built on a fully materialized plan indexed by
`ledger_offset`, etc.) is in
[`openspec/changes/session6-data-execution-system/design.md`](../openspec/changes/session6-data-execution-system/design.md).

## Key design decisions

- **No real tokenizer/model.** The assignment explicitly allows a small
  tokenizer at toy scale; a frozen whitespace vocab with a verifiable
  `sha256` hash demonstrates the same integrity guarantee without a
  dependency. Toy loss is a deterministic function of step/lane/phase
  (perplexity-phase multipliers mined from the course material), not a real
  forward pass.
- **Hash-seeded OPUS scoring, not `random`.** Reproducibility across
  crash/resume/replay boundaries needs a pure function of `candidate_batch_id`
  — no RNG state to lose when the process "crashes". One candidate batch gets
  exactly one verdict: re-rolling the id until the hash cleared the threshold
  would make the reject rate an artifact of the retry count instead of a
  property of the stream.

- **The OPUS proxy score is independent of the lane's mixture weight.**
  `pick_lane` already samples the lane by its profile share; scoring by that
  same share again double-counts the mixture and re-filters an
  already-correct stream. Blending it in (`0.6*weight + 0.4*jitter` against a
  `0.6` threshold) made acceptance impossible for any lane under a 1/3 share,
  so every lane but `general` could only enter training through the
  protected-floor override, and the realized stream collapsed to 86% general
  against a planned 45%. OPUS filters on quality; the mixture is applied
  once, upstream.
- **One shard = one document; `docs_per_batch=3` documents per candidate
  batch.** One shard per document lets the eval firewall's per-shard fields
  (`never_train`, `benchmark_overlap_pct`, ...) attach directly. Candidate
  batches then draw *several* shards, so segment-id attention blocking and
  per-document position resets are exercised by the real committed stream
  (81 of 191 bins are multi-document) rather than only by a unit test. A
  candidate batch can produce more than one bin: `structure_preserving` and
  `long_context` deliberately refuse to merge documents.

- **The protected-floor override is recorded whenever the floor steers lane
  selection**, not only when it rescues a candidate the proxy would have
  dropped. Under the current profiles Session 5's 8% combined floor is slack
  (planned shares are 11.7/15.3/8.3 against floors of 3/3/2), so it binds 3
  times in 150 steps — all during warmup — and never has to rescue anything.
  The rescue path itself is pinned by `test_opus.py` rather than left to
  chance in the artifacts.

- **The eval firewall gates admission, not just selection.** A manifest whose
  shard trips the firewall is scored at the hard-block cap and comes out
  `admission="blocked"`. Relying on the training loop simply never picking
  those shards would make firewall correctness an accident of an unrelated
  loop bound.

- **Contamination, PII and duplication are measured, not declared** — using
  Session 4's cleaning pipeline (`cleaning.py`, vendored). `benchmark_overlap_pct`
  is a real 8-gram fingerprint overlap against the eval set, `canary_match` is
  a real string search, `pii_screen_status` comes from Session 4's structured-
  identifier regexes, and `dedup_status` from Jaccard shingle comparison. These
  were previously constants in the corpus and default arguments on
  `build_manifest`, which meant 67 manifests all asserted `dedup_status="passed"`
  with no dedup code anywhere in the repo.

  For the detectors to measure anything, the corpus has to *contain* something:
  it now plants verbatim eval spans in three training documents, a canary
  string, structured PII, and two near-duplicates. `test_cleaning.py` fails if
  those defects ever stop being detected — the failure mode being guarded
  against is the fields silently reverting to measurements of nothing.

  Provenance flags (`never_train`, `benchmark_derived`) stay declared, because
  "this text was derived from a benchmark" is not detectable from the text —
  it comes from the Session 3 source contract.

- **PII is masked before the vocabulary is built.** Masking only before
  tokenization still leaks: a tokenizer trained on raw text bakes the
  identifiers into `tokenizer.json`, which is a published artifact. Caught by
  `test_pii_never_reaches_a_token_id`.

- **Ledger token spans are real document spans.** `token_span_ids` records
  `shard_id:start:end` into the shard's own token stream (its tokens plus a
  trailing EOS), never the padded sample length. Replay slices those spans
  straight back out of the content-addressed shard instead of re-running the
  packer, so it reproduces the exact fragment even when `concat_and_chop`
  split one document across two bins — and a wrong span produces a different
  hash instead of being quietly re-derived.

- **The crash lands between checkpoints, and resume rolls the ledgers back.**
  A crash that stops exactly on a checkpoint boundary proves nothing: nothing
  is in flight, so "no repeated batches" holds trivially. The drill crashes
  mid-interval, truncates both ledgers back to the last durable checkpoint
  (`JsonlLedger.truncate_after`), and replays the rolled-back batches exactly
  once.
- **Materialized plan indexed by `ledger_offset`.** The entire mixture→OPUS→
  packing plan is computed once as an ordered list; crash/resume becomes
  "reload the checkpoint's offset, assert `plan[offset]` is next," and
  ledger-mode replay becomes "recompute from shards, compare hashes." Both
  invariants become direct assertions instead of emergent dataloader
  behavior. This doesn't scale to a real run (`ponytail: holds the whole
  plan in memory; switch to a streaming dataloader if this needs to scale
  past a toy corpus`), which is fine — the assignment is explicit that scale
  isn't the goal.
- **Attention as segment ids, not a dense mask.** Real packed-sequence
  implementations (Megatron, FlashAttention varlen) use segment ids /
  `cu_seqlens`, not a materialized seq_len × seq_len matrix — so `batch.py`
  does the same.

## Reproduce

```bash
python Session6/run_demo.py                      # regenerates submission_artifacts/
python -m pytest Session6/pipeline/tests/ -q      # 74 tests
```

## Results (measured, from a real `submission_artifacts/` run)

| Check | Result |
|---|---|
| Tokenizer integrity | **PASS** — 67 manifests verified against `manifests/tokenizer.json` |
| Evaluation firewall | **PASS** — 11 firewall-blocked shards (7 eval candidates + 4 contaminated training docs), all `admission="blocked"`, 0 leaked into training |
| Data cleaning | **PASS** — measured, not declared: 10 shards overlapping the eval set, 1 canary hit, 2 shards with PII masked (4 identifiers), 2 near-duplicates, all blocked |
| Packing correctness | **PASS** — all 191 packed samples re-verified (mask hashes, segment ids, position resets, span lengths); 81 multi-document; 41.2% utilization recomputed from the ledger's spans and matching the reported number |
| Mixture compliance | **PASS** — realized shares track the stage-weighted plan within ±8 points; all protected floors met |
| OPUS audit trail | **PASS** — all 150 candidate decisions logged (113 accepted, 7 deferred, 30 rejected, each with a reason) plus 3 protected-floor overrides |
| Crash recovery | **PASS** — crashed mid-interval at offset 14, rolled 5 in-flight events back to the offset-9 checkpoint, resumed at 10, no batch skipped or repeated |
| Replay | **PASS** — ledger-mode replay of offsets [0, 10) reproduced identical shard ids, token spans and mask hashes |
| Learning trace | **PASS** — all 353 learning-ledger rollups trace to a real manifest shard id |
| Throughput | **PASS** — 21,902 useful tokens/sec, fully recomputable from `performance.json`'s stored `inputs` |

**On the 42.24% packing utilization.** It is low, and honestly so — the
throughput report's own `advice` field says `packing is the bottleneck`. The
breakdown by policy shows exactly where it goes: `concat_and_chop` fills
96.09% of its bins and `best_fit` 93.75%, while `long_context` fills 15.62%
because it reserves a 256-token window for a ~40-token toy reasoning
document, and `structure_preserving` fills 22.29% by refusing to merge
agentic turns. That is the packing/throughput lesson working as intended
rather than a number to tune away: policies chosen for correctness cost
utilization, and the report says so.

Every row is re-derived from the generated files by `evidence.py`. The
packing, firewall, OPUS and crash rows are covered by tamper tests
(`test_evidence.py`) that corrupt an artifact and assert the row flips to
FAIL, so a PASS is not a tautology.

Full evidence table: `submission_artifacts/evidence.md` (generated fresh by
`evidence.py` each run — never hand-edited).

## Known ceilings (deliberate, toy-scale simplifications)

- The whole run plan is held in memory rather than streamed — fine at this
  scale, would need a real dataloader to go further.
- The toy tokenizer isn't a real BPE — swappable later since `tokenizer.py`
  isolates the interface.
- `useful_tokens_per_sec` uses the course's own formula sketch fed with real
  measured packing/reject rates, not a benchmarked real-hardware number.
- `loss_delta` in the learning ledger is still `0.0` — the toy loss is a
  function of step/lane/phase, so a per-shard delta would measure the phase
  schedule, not learning.
- `long_context` reserves 256 tokens per bin against ~40-token toy documents,
  so it dominates the utilization shortfall; at real reasoning-trace lengths
  the same policy would fill its window.
- `feasibility_check` reports scarcity warnings but no upsample/relax/accept
  policy acts on them.
- Shards live in memory and are re-derived deterministically; only manifests
  are written to disk.
