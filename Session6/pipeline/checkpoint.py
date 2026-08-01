"""Checkpoint records tied to a ledger offset.

`latest_checkpoint` tracks the most recent save via an append-only index
file rather than filesystem mtimes, so it stays correct even when saves
happen faster than the filesystem's mtime resolution.
"""
import hashlib
import json
import os


def _hash_state(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def build_checkpoint(checkpoint_id, global_step, ledger_offset, run_branch_id,
                      model_state, optimizer_state, rng_state):
    return {
        "checkpoint_id": checkpoint_id,
        "global_step": global_step,
        "ledger_offset": ledger_offset,
        "run_branch_id": run_branch_id,
        "model_state_hash": _hash_state(model_state),
        "optimizer_state_hash": _hash_state(optimizer_state),
        "rng_state": rng_state,
        "dataloader_state": f"ledger_offset_{ledger_offset}",
    }


def save_checkpoint(checkpoint, directory):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{checkpoint['checkpoint_id']}.json")
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    with open(os.path.join(directory, "index.jsonl"), "a") as f:
        f.write(json.dumps({"checkpoint_id": checkpoint["checkpoint_id"],
                             "global_step": checkpoint["global_step"],
                             "ledger_offset": checkpoint["ledger_offset"]}) + "\n")
    return path


def load_checkpoint(directory, checkpoint_id):
    with open(os.path.join(directory, f"{checkpoint_id}.json")) as f:
        return json.load(f)


def _read_index(directory):
    index_path = os.path.join(directory, "index.jsonl")
    if not os.path.exists(index_path):
        return []
    with open(index_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_checkpoint(directory):
    lines = _read_index(directory)
    if not lines:
        return None
    return load_checkpoint(directory, lines[-1]["checkpoint_id"])


def first_checkpoint(directory):
    lines = _read_index(directory)
    if not lines:
        return None
    return load_checkpoint(directory, lines[0]["checkpoint_id"])
