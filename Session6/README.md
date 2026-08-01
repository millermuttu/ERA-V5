# Session 6 — Training Data Execution System (ERA V5)

A small but complete implementation of the full training-data path:
`documents -> tokenized shards -> manifests -> mixture schedule -> packing ->
batches -> training -> consumption ledger -> learning ledger -> checkpoint ->
crash -> resume -> replay -> fork -> audit`, at toy scale, proving the system
is correct, reproducible, and auditable rather than proving it scales.

**Deliverable:** `python run_demo.py` — regenerates `submission_artifacts/`
from scratch on every run.

## Architecture

17 focused, independently-tested modules under `pipeline/`, mirroring
Session 4's pipeline/tests layout:

| Module | Responsibility |
|---|---|
| `corpus.py` | Deterministic seeded toy corpus: 5 training lanes (general/code/indic/reasoning/agentic) + 7 eval/benchmark candidates |
| `tokenizer.py` | Frozen whitespace/punctuation tokenizer with a `sha256` vocab hash (stdlib only — a real BPE isn't needed at this scale) |
| `shards.py` | Content-addressed, immutable shards (`shard_id`/`content_hash` from tokenized content) |
| `manifest.py` | Shard manifest schema + admission-gate scoring (hard-required fields, cap-at-64 on hard-block, admit threshold > 86) |
| `firewall.py` | Single canonical 4-gate eval/validation firewall, called at admission *and* batch assembly |
| `mixture.py` | Curriculum stage bands (Foundation/Skill-build/Anneal), lane profiles, protected-floor enforcement |
| `opus.py` | Deterministic (hash-seeded) admission scoring: accepted/rejected/deferred/protected |
| `packing.py` | 6 packing policies (pad_only, concat_and_chop, greedy, best_fit, structure_preserving, long_context) |
| `batch.py` | Loss mask / segment-id attention blocking / position-id reset per document |
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
  — no RNG state to lose when the process "crashes".
- **One shard = one document.** Simpler than grouping docs per lane, and lets
  the eval firewall's per-shard fields (`never_train`, `benchmark_overlap_pct`,
  ...) attach directly.
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
python -m pytest Session6/pipeline/tests/ -q      # 45 tests
```

## Results (measured, from a real `submission_artifacts/` run)

| Check | Result |
|---|---|
| Tokenizer integrity | **PASS** — 67 manifests verified against `manifests/tokenizer.json` |
| Evaluation firewall | **PASS** — 5 of 7 eval candidates blocked, 0 leaked into training |
| Packing correctness | **PASS** — 37.04% packing utilization (toy docs are short relative to `seq_len=128`) |
| Mixture compliance | **PASS** — indic/reasoning/agentic all held at/above their protected floor |
| OPUS audit trail | **PASS** — 43 accepted, 6 protected-floor overrides, 34 deferred, 67 rejected |
| Crash recovery | **PASS** — crashed at offset 9, resumed at offset 10, next batch matched exactly |
| Replay | **PASS** — ledger-mode replay of offsets [0, 10) reproduced identical hashes |
| Learning trace | **PASS** — every learning-ledger rollup traces to a real manifest shard id |
| Throughput | **PASS** — 8,538 useful tokens/sec, fully recomputable from `performance.json`'s stored `inputs` |

Full evidence table: `submission_artifacts/evidence.md` (generated fresh by
`evidence.py` each run — never hand-edited).

## Known ceilings (deliberate, toy-scale simplifications)

- The whole run plan is held in memory rather than streamed — fine at this
  scale, would need a real dataloader to go further.
- The toy tokenizer isn't a real BPE — swappable later since `tokenizer.py`
  isolates the interface.
- `useful_tokens_per_sec` uses the course's own formula sketch fed with real
  measured packing/reject rates, not a benchmarked real-hardware number.
