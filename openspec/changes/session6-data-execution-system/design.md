## Context

See `proposal.md` - Why/What Changes for motivation and scope. Constraints that shape this design:

- Toy scale is explicitly licensed by the assignment ("may use a small corpus, tokenizer and model... the goal is not scale") - this design avoids real BPE, real GPUs, and real models entirely, using stdlib-only Python.
- The repo already has a working precedent for a code-deliverable session: `Session4/pipeline/` (pure, independently-tested functions), `Session4/pipeline/tests/`, `Session4/README.md` (title, strategy list, results table, exact reproduce commands), and `Session4/pipeline/conftest.py` (`sys.path.insert` so tests import the package without installation). Session6 reuses this shape.
- `Session6/reference/mined-numbers.md` and `session6-lesson.txt` already define concrete JSON shapes, thresholds, and formulas (manifest admission scoring, mixture stage bands, ledger event shapes, OPUS decision taxonomy, firewall gates, replay offset-shift semantics, throughput formulas) mined from the course's 15 widgets - this design reuses those shapes rather than inventing new ones.
- Session5's `submission.md` already defines the lane names and stage percentages for the mixture/curriculum plan; Session6 reuses these as defaults rather than re-deriving a new mixture design.
- No `openspec/specs/` capabilities exist yet in this repo, so `data-execution-system` is a standalone new capability with no cross-capability interactions to reconcile.

## Goals / Non-Goals

**Goals:**
- One command (`python run_demo.py`) exercises the entire path from documents to audit and produces the exact `submission_artifacts/` tree the assignment requires.
- Every invariant the grading rubric checks (packing/masks, mixture/OPUS, ledgers, checkpoint/crash/resume/replay/fork, firewall, throughput) is backed by a real generated artifact and a corresponding pytest test.
- The evidence bundle is computed by scanning generated artifacts - it must be possible to break an invariant and see the evidence bundle reflect a real FAIL, never a hardcoded PASS.
- Runs are fully deterministic and reproducible: fixed seeds, hash-derived "randomness" for OPUS scoring, no wall-clock-sensitive assertions.

**Non-Goals:**
- No real tokenizer (no BPE/tiktoken), no real model, no real GPU/distributed training - all "training" is a deterministic toy loss computation over packed token batches.
- No network fetch or external dataset download (unlike Session4) - the corpus is generated in-process, deterministically seeded.
- No production-grade performance; throughput numbers are illustrative and reconstructable, not benchmarked against real hardware.
- Does not modify Session1-5 code; only reuses Session5's mixture lane percentages as reference constants.

## Decisions

**Module layout under `Session6/pipeline/`** (mirrors `Session4/pipeline/` conventions): `corpus.py`, `tokenizer.py`, `shards.py`, `manifest.py`, `firewall.py`, `mixture.py`, `opus.py`, `packing.py`, `batch.py`, `ledger.py`, `checkpoint.py`, `train_loop.py`, `crash_resume.py`, `replay.py`, `audit.py`, `throughput.py`, `evidence.py`, each with a matching `tests/test_*.py`, plus `conftest.py` for `sys.path` setup.
- *Why*: one focused module per pipeline stage keeps each piece independently testable, matching the precedent Session4 already established in this repo, rather than one large script.
- *Alternative considered*: a single monolithic `pipeline.py`. Rejected - the rubric grades each stage's correctness separately (packing/masks, ledgers, checkpoint/resume, etc.), so per-stage modules make it easy to point evidence and tests at a specific file.

**Toy tokenizer instead of real BPE.**
- *Why*: the assignment explicitly allows a small tokenizer; a whitespace/punctuation vocab built from the generated corpus, hashed with `sha256`, is sufficient to demonstrate "frozen tokenizer and content hashes" without pulling in a dependency or reusing Session2's much larger BPE tokenizer, which was built for a different corpus and would add coupling for no benefit.
- *Alternative considered*: reuse Session2's BPE tokenizer. Rejected for now - it would tie Session6's reproducibility to Session2's artifacts and add complexity the rubric doesn't reward; can be swapped in later since `tokenizer.py` isolates the interface (`ponytail: swap in Session2's BPE only if a real tokenizer becomes graded`).

**Content-addressed, immutable shards.**
- *Why*: `shard_id`/`content_hash` derived from `sha256` of tokenized content directly satisfies "immutable tokenized shards" and makes the idempotency requirement (identical input -> identical id) trivially testable.

**Deterministic, hash-seeded OPUS scoring instead of `random`.**
- *Why*: the rubric requires reproducibility ("prove that the next batch is exactly the expected batch"); using `int(sha256(candidate_id).hexdigest(), 16)` as a pseudo-random jitter source makes scoring a pure function of `candidate_batch_id`, so two runs (or a resume) always agree.
- *Alternative considered*: Python's `random` module with a fixed seed. Rejected - reseeding correctly across crash/resume/replay/fork boundaries is fragile and easy to get subtly wrong; a hash-of-id function has no state to lose.

**Single canonical firewall function called at two sites.**
- *Why*: the rubric explicitly penalizes eval data leaking into a loss-bearing batch, and calling one `firewall.check(shard)` function both at shard-admission time and again as an assertion inside batch assembly gives defense-in-depth without duplicating the four-gate logic in two places (a change to the gate logic only has one place to go wrong).

**Packing policies mapped by lane, with `long_context` implemented as a tagged variant of `pad_only`.**
- *Why*: the assignment lists distinct named policies (pad-only, concat-and-chop, greedy, best-fit, structure-preserving, long-context) but `long_context` in the course material is really `pad_only` with a larger window - implementing it as a thin variant avoids a duplicate near-identical bin-packing implementation. Marked `ponytail: shares pad_only's implementation; split only if long-context ever needs real cross-window logic.`
- Masks follow Megatron-style packed-sequence semantics: `position_ids` reset per document, `attention_mask` blocked across document boundaries, `loss_mask` excludes padding and non-loss-bearing roles (e.g. agentic `user`/`observation` turns) - this is the standard, well-tested approach for packed multi-document batches and is what the rubric's "correct loss masks, attention masks and position ids" line is checking for.

**Mixture stage determined by cumulative tokens consumed / configured toy budget, not step count.**
- *Why*: tying stage transitions to real consumed-token fractions (Foundation 0-55%, Skill-build 55-85%, Anneal 85-100%, reusing Session5's bands) is more auditable than a magic step threshold, and ties directly to the ledger's own recorded token counts rather than a separately-maintained counter that could drift out of sync.

**Checkpoint/resume/replay built on a fully materialized batch plan indexed by `ledger_offset`.**
- *Why*: precomputing the entire mixture -> OPUS -> packing plan once (a list of batch records) before "training" starts means `ledger_offset` is simply an index into this list - crash/resume becomes "reload the last checkpoint's offset, assert `plan[offset]` is the next batch processed," and ledger-mode replay becomes "recompute batch content from `plan[start:end]` and shard data, compare hashes." This makes both invariants ("no skip/repeat" and "hash equality") direct, testable assertions rather than emergent behavior of a stateful streaming loader.
- *Alternative considered*: a streaming, on-the-fly dataloader (closer to a real system, closer to Mosaic StreamingDataset). Rejected for this toy scale - it would require reproducing shuffle-buffer state across crash/resume/replay, adding real complexity the rubric doesn't require proving beyond correctness of the three invariants above.
- Replay supports 3 modes matching the course material's offset-shift semantics: `ledger` (exact reconstruction, asserted equal to the original - this is the one the rubric requires to pass), `random` (offset +9, no ledger binding - logged as a deliberately non-matching negative example, not claimed as a passing invariant), `fork` (offset +17, new `run_branch_id` - used for the fork-from-checkpoint requirement).

**Throughput formulas reused verbatim from `mined-numbers.md` (widget 15), but computed from real measured run stats rather than the widget's example slider defaults.**
- *Why*: the rubric explicitly penalizes "unreconstructable throughput numbers" - storing the formula `inputs` alongside the outputs in `performance.json` lets a grader (or `test_throughput.py`) recompute and verify the output independently.

**Evidence bundle generated by scanning artifacts, never by writing literal PASS/FAIL strings inline in `run_demo.py`.**
- *Why*: the assignment explicitly states "hardcoded evidence will not be accepted" and graders will "inspect the code to confirm evidence wasn't simulated." `evidence.py` reads the generated `manifests/`, `ledgers/`, `checkpoints/`, and `performance.json` files and derives each row's result from their actual contents, so breaking an invariant upstream necessarily flips the corresponding evidence row.

## Risks / Trade-offs

- **[Risk]** Toy scale may look "too simple" to a grader skimming for realism. -> **Mitigation**: keep config field *names* aligned with the real-scale defaults (gpus/micro_batch/grad_accum/seq_len/ckpt_interval), just with smaller values, so the shape of a real training config is visible even though absolute numbers are toy-sized.
- **[Risk]** Hash-seeded "determinism" could accidentally look like it's gaming the reproducibility requirement rather than genuinely computing it. -> **Mitigation**: `test_opus.py` and `test_replay.py` assert equality across independently invoked runs (not just within one process), and the design doc/README explain the hash-seeding rationale explicitly.
- **[Risk]** A fully materialized batch plan (holding the whole run's batch list in memory) doesn't scale to a real training run. -> **Mitigation**: acceptable at toy scale per the assignment's explicit "goal is not scale" framing; noted as a known ceiling in the README rather than hidden.
- **[Risk]** Skipping a real BPE tokenizer could be seen as under-delivering on "frozen tokenizer" fidelity. -> **Mitigation**: the tokenizer still produces a genuine frozen hash and a genuine vocabulary artifact; `tokenizer.py`'s interface is isolated so a real BPE could be substituted later without touching downstream modules.

## Open Questions

None - toy-scale scope, module boundaries, and algorithm choices are all settled above; nothing here needs to change the specs, approach, or task breakdown.
