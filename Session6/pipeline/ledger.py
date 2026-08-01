"""Append-only JSONL consumption ledger + learning ledger.

Event shapes mined from the course's training-consumption-ledger and
shard-learning-report-card widgets (`Session6/reference/mined-numbers.md`,
widgets 8 and 12).
"""
import json
import os


class JsonlLedger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            open(path, "w").close()

    def append(self, event):
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def read_all(self):
        events = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def truncate_after(self, ledger_offset):
        """Crash recovery: drop everything written past the last durable
        checkpoint. Appends between a checkpoint and a crash were never
        covered by a fsync'd checkpoint, so on resume they must not be
        counted as consumed - otherwise those batches are trained twice.
        Returns the number of dropped events."""
        events = self.read_all()
        kept = [e for e in events if e.get("ledger_offset", -1) <= ledger_offset]
        with open(self.path, "w") as f:
            for event in kept:
                f.write(json.dumps(event) + "\n")
        return len(events) - len(kept)


def batch_committed_event(ledger_offset, run_branch_id, global_step, sample, opus_decision,
                           mixture_lane, curriculum_stage, rank=0, microbatch_count=1,
                           checkpoint_id=None, created_at=""):
    return {
        "event": "batch_committed",
        "ledger_offset": ledger_offset,
        "run_branch_id": run_branch_id,
        "global_step": global_step,
        "checkpoint_id": checkpoint_id,
        "created_at": created_at,
        "rank": rank,
        "microbatch_count": microbatch_count,
        "packed_sample_ids": [f"sample-{ledger_offset}"],
        "shard_ids": sample["shard_ids"],
        # Real per-document spans into each shard's token stream - NOT the padded
        # sample length. Replay slices the shard back out using exactly these.
        "token_span_ids": sample["token_span_ids"],
        "seq_len": sample["seq_len"],
        "loss_mask_hash": sample["loss_mask_hash"],
        "position_policy": sample["policy"],
        "mixture_lane": mixture_lane,
        "curriculum_stage": curriculum_stage,
        "opus_decision_id": opus_decision["opus_decision_id"],
    }


def checkpoint_bound_event(checkpoint_id, ledger_offset, run_branch_id, model_state_hash,
                            optimizer_state_hash, rng_state, created_at=""):
    return {
        "event": "checkpoint_bound",
        "checkpoint_id": checkpoint_id,
        "ledger_offset": ledger_offset,
        "run_branch_id": run_branch_id,
        "model_state": model_state_hash,
        "optimizer_state": optimizer_state_hash,
        "dataloader_state": f"ledger_offset_{ledger_offset}",
        "rng_state": rng_state,
        "created_at": created_at,
    }


def learning_ledger_event(shard_id, lane, stage_phase, avg_token_loss, loss_delta, gradient_norm,
                           opus_score, tokens_consumed, global_step, ledger_offset, repeated_pass=0,
                           checkpoint_before=None, checkpoint_after=None):
    if avg_token_loss < 2.0:
        usefulness = "useful"
    elif avg_token_loss < 4.0:
        usefulness = "review"
    else:
        usefulness = "delay"
    return {
        "event": "learning_rollup",
        "ledger_offset": ledger_offset,
        "shard_id": shard_id,
        "lane": lane,
        "stage_phase": stage_phase,
        "avg_token_loss": avg_token_loss,
        "loss_delta": loss_delta,
        "gradient_norm": gradient_norm,
        "opus_score": opus_score,
        "repeated_pass": repeated_pass,
        "tokens_consumed": tokens_consumed,
        "usefulness_classification": usefulness,
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "global_step": global_step,
    }
