"""Deliberate mid-stream crash + resume, proving no batch is skipped or
repeated across the boundary.

`checkpoint.latest_checkpoint` is read from disk (not from in-process
memory), so the resume genuinely depends on what got persisted, not on
remembering where the crash happened.
"""
from pipeline.checkpoint import latest_checkpoint
from pipeline.train_loop import run_plan


def simulate_crash_and_resume(plan, config, ledger, learning_ledger, checkpoint_dir, run_branch_id):
    """Runs up through the first checkpoint boundary, "crashes" (the loop
    simply stops), then resumes purely from the checkpoint found on disk."""
    crash_offset = config.ckpt_interval - 1
    run_plan(plan, config, ledger, learning_ledger, checkpoint_dir, run_branch_id,
              start=0, end=crash_offset + 1)

    # --- crash: nothing beyond the last checkpoint is durable ---

    checkpoint = latest_checkpoint(checkpoint_dir)
    resume_offset = checkpoint["ledger_offset"] + 1

    run_plan(plan, config, ledger, learning_ledger, checkpoint_dir, run_branch_id,
              start=resume_offset, end=len(plan))

    return {"crash_offset": crash_offset, "resume_offset": resume_offset,
            "checkpoint_id": checkpoint["checkpoint_id"]}


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
            and actual["loss_mask_hash"] == expected["loss_mask_hash"])
