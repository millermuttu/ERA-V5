from pipeline.ledger import JsonlLedger
from pipeline.replay import replay_fork_mode, replay_ledger_mode, replay_random_mode
from pipeline.train_loop import Config, build_world, materialize_plan, run_plan


def _populated_ledger(tmp_path):
    config = Config(total_steps=200, ckpt_interval=10)
    world = build_world(config)
    plan = materialize_plan(world, config)
    ledger = JsonlLedger(str(tmp_path / "consumption.jsonl"))
    learning_ledger = JsonlLedger(str(tmp_path / "learning.jsonl"))
    checkpoint_dir = str(tmp_path / "checkpoints")
    run_plan(plan, config, ledger, learning_ledger, checkpoint_dir, "run-a")
    return ledger, plan, world, config


def test_ledger_mode_replay_hashes_match(tmp_path):
    ledger, plan, world, config = _populated_ledger(tmp_path)
    assert len(plan) >= 10
    results = replay_ledger_mode(ledger, world["training_shards"], world["tokenizer"], config, 0, 10)
    assert all(r["matched"] for r in results)


def test_random_mode_offset_shift_formula():
    result = replay_random_mode(start=5)
    assert result["shifted_start"] == 14
    assert result["ledger_bound"] is False


def test_fork_mode_offset_shift_and_branch_change():
    result = replay_fork_mode(checkpoint_offset=20, original_run_branch_id="run-a")
    assert result["new_start_offset"] == 37
    assert result["run_branch_id"] == "run-b"
