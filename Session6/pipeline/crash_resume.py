"""Deliberate mid-stream crash + resume, proving no batch is skipped or
repeated across the boundary.

The crash lands *between* checkpoints on purpose. A crash that happens to
land exactly on a checkpoint boundary proves nothing: nothing was in flight,
so "no repeated batches" holds trivially. The real invariant is that batches
committed after the last durable checkpoint get rolled back on resume and
then replayed exactly once.

`checkpoint.latest_checkpoint` is read from disk (not from in-process
memory), so the resume genuinely depends on what got persisted, not on
remembering where the crash happened.
"""
from pipeline.checkpoint import latest_checkpoint
from pipeline.train_loop import run_plan


def default_crash_offset(config):
    """Mid-interval: half a checkpoint interval past the first checkpoint."""
    return config.ckpt_interval + max(1, config.ckpt_interval // 2) - 1


def simulate_crash_and_resume(plan, config, ledger, learning_ledger, checkpoint_dir, run_branch_id,
                               crash_offset=None, opus_ledger=None):
    """Runs past the first checkpoint into the middle of the next interval,
    "crashes" (the loop simply stops), rolls the ledgers back to the last
    durable checkpoint, then resumes purely from what is on disk."""
    crash_offset = default_crash_offset(config) if crash_offset is None else crash_offset
    crash_offset = min(crash_offset, len(plan) - 1)
    run_plan(plan, config, ledger, learning_ledger, checkpoint_dir, run_branch_id,
              start=0, end=crash_offset + 1, opus_ledger=opus_ledger)

    # --- crash: nothing past the last checkpoint was ever durable ---
    checkpoint = latest_checkpoint(checkpoint_dir)
    rolled_back = ledger.truncate_after(checkpoint["ledger_offset"])
    learning_ledger.truncate_after(checkpoint["ledger_offset"])

    resume_offset = checkpoint["ledger_offset"] + 1
    run_plan(plan, config, ledger, learning_ledger, checkpoint_dir, run_branch_id,
              start=resume_offset, end=len(plan), opus_ledger=opus_ledger)

    return {"crash_offset": crash_offset, "resume_offset": resume_offset,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "in_flight_events_rolled_back": rolled_back,
            "batches_replayed_after_rollback": crash_offset - checkpoint["ledger_offset"]}


def verify_no_skip_no_repeat(ledger, plan_length):
    offsets = [e["ledger_offset"] for e in ledger.read_all() if e["event"] == "batch_committed"]
    return sorted(offsets) == list(range(plan_length)) and len(offsets) == len(set(offsets))


def verify_resume_matched_expected(ledger, plan, resume_offset):
    events_by_offset = {e["ledger_offset"]: e for e in ledger.read_all() if e["event"] == "batch_committed"}
    actual = events_by_offset.get(resume_offset)
    if actual is None:
        return False
    expected = plan[resume_offset]["sample"]
    return (actual["shard_ids"] == expected["shard_ids"]
            and actual["token_span_ids"] == expected["token_span_ids"]
            and actual["loss_mask_hash"] == expected["loss_mask_hash"])
