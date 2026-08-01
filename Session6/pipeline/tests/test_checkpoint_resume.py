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


def _crash_at(tmp_path, crash_offset):
    from pipeline.crash_resume import simulate_crash_and_resume
    from pipeline.ledger import JsonlLedger
    from pipeline.train_loop import Config, build_world, materialize_plan

    config = Config(total_steps=200, ckpt_interval=5)
    plan = materialize_plan(build_world(config), config)
    assert len(plan) > 15

    ledger = JsonlLedger(str(tmp_path / "consumption.jsonl"))
    learning_ledger = JsonlLedger(str(tmp_path / "learning.jsonl"))
    result = simulate_crash_and_resume(plan, config, ledger, learning_ledger,
                                        str(tmp_path / "checkpoints"), "run-a",
                                        crash_offset=crash_offset)
    return result, ledger, learning_ledger, plan


def test_crash_resume_no_skip_no_repeat(tmp_path):
    from pipeline.crash_resume import verify_no_skip_no_repeat, verify_resume_matched_expected

    result, ledger, _, plan = _crash_at(tmp_path, crash_offset=None)
    assert verify_no_skip_no_repeat(ledger, len(plan))
    assert verify_resume_matched_expected(ledger, plan, result["resume_offset"])


def test_default_crash_lands_between_checkpoints(tmp_path):
    """A crash exactly on a checkpoint boundary proves nothing - there is
    nothing in flight to roll back, so the invariant holds trivially."""
    from pipeline.train_loop import Config
    from pipeline.crash_resume import default_crash_offset

    config = Config(ckpt_interval=5)
    assert (default_crash_offset(config) + 1) % config.ckpt_interval != 0

    result, _, _, _ = _crash_at(tmp_path, crash_offset=None)
    assert result["batches_replayed_after_rollback"] > 0
    assert result["in_flight_events_rolled_back"] > 0


def test_crash_mid_interval_does_not_repeat_batches(tmp_path):
    """The regression this whole module exists for: batches committed after
    the last checkpoint must be rolled back, not trained a second time."""
    from pipeline.crash_resume import verify_no_skip_no_repeat

    result, ledger, learning_ledger, plan = _crash_at(tmp_path, crash_offset=13)
    assert result["resume_offset"] == 10, "should resume from the offset-9 checkpoint"
    assert result["batches_replayed_after_rollback"] == 4

    offsets = [e["ledger_offset"] for e in ledger.read_all() if e["event"] == "batch_committed"]
    assert len(offsets) == len(set(offsets)), f"duplicated offsets: {sorted(offsets)}"
    assert verify_no_skip_no_repeat(ledger, len(plan))

    learn_offsets = [e["ledger_offset"] for e in learning_ledger.read_all()]
    assert sorted(set(learn_offsets)) == list(range(len(plan)))
