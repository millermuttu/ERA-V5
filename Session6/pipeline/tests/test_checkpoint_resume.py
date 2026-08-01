from pipeline.checkpoint import build_checkpoint, latest_checkpoint, save_checkpoint


def test_save_and_load_roundtrip(tmp_path):
    ckpt = build_checkpoint("ckpt-1", global_step=5, ledger_offset=10, run_branch_id="run-a",
                             model_state={"w": 1}, optimizer_state={"m": 1}, rng_state="rng-a")
    save_checkpoint(ckpt, str(tmp_path))
    loaded = latest_checkpoint(str(tmp_path))
    assert loaded["checkpoint_id"] == "ckpt-1"
    assert loaded["ledger_offset"] == 10


def test_latest_checkpoint_tracks_most_recent(tmp_path):
    ckpt1 = build_checkpoint("ckpt-1", 5, 10, "run-a", {}, {}, "rng-a")
    ckpt2 = build_checkpoint("ckpt-2", 10, 20, "run-a", {}, {}, "rng-b")
    save_checkpoint(ckpt1, str(tmp_path))
    save_checkpoint(ckpt2, str(tmp_path))
    latest = latest_checkpoint(str(tmp_path))
    assert latest["checkpoint_id"] == "ckpt-2"


def test_crash_resume_no_skip_no_repeat(tmp_path):
    from pipeline.crash_resume import (simulate_crash_and_resume, verify_no_skip_no_repeat,
                                        verify_resume_matched_expected)
    from pipeline.ledger import JsonlLedger
    from pipeline.train_loop import Config, build_world, materialize_plan

    config = Config(total_steps=200, ckpt_interval=5)
    world = build_world(config)
    plan = materialize_plan(world, config)
    assert len(plan) > 15

    ledger = JsonlLedger(str(tmp_path / "consumption.jsonl"))
    learning_ledger = JsonlLedger(str(tmp_path / "learning.jsonl"))
    checkpoint_dir = str(tmp_path / "checkpoints")

    result = simulate_crash_and_resume(plan, config, ledger, learning_ledger, checkpoint_dir, "run-a")

    assert verify_no_skip_no_repeat(ledger, len(plan))
    assert verify_resume_matched_expected(ledger, plan, result["resume_offset"])
