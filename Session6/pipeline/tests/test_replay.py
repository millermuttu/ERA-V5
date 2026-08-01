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


def test_token_spans_are_real_not_padded_length(tmp_path):
    """The ledger used to record `shard:0:<padded seq_len>` for every doc."""
    ledger, plan, world, config = _populated_ledger(tmp_path)
    counts = {s["shard_id"]: s["token_count"] for s in world["training_shards"]}
    events = [e for e in ledger.read_all() if e["event"] == "batch_committed"]

    for event in events:
        assert len(event["token_span_ids"]) == len(event["shard_ids"])
        for span in event["token_span_ids"]:
            shard_id, start, end = span.rsplit(":", 2)
            start, end = int(start), int(end)
            assert start < end
            # +1 for the EOS concat_and_chop appends at index token_count
            assert end <= counts[shard_id] + 1
    assert any(int(s.rsplit(":", 2)[2]) != e["seq_len"]
               for e in events for s in e["token_span_ids"]), "spans still look like padded lengths"


def test_replay_detects_a_tampered_span(tmp_path):
    """Replay must reconstruct from the recorded spans, so a wrong span has
    to produce a different hash rather than being quietly re-derived."""
    ledger, plan, world, config = _populated_ledger(tmp_path)
    event = dict(next(e for e in ledger.read_all() if e["event"] == "batch_committed"))
    shard_id, start, end = event["token_span_ids"][0].rsplit(":", 2)
    event["token_span_ids"] = [f"{shard_id}:{start}:{int(end) - 3}"] + event["token_span_ids"][1:]

    from pipeline.replay import reconstruct_sample
    rebuilt = reconstruct_sample(event, world["training_shards"], world["tokenizer"], config)
    assert rebuilt["loss_mask_hash"] != event["loss_mask_hash"]


def test_multi_document_bins_replay_exactly(tmp_path):
    ledger, plan, world, config = _populated_ledger(tmp_path)
    results = replay_ledger_mode(ledger, world["training_shards"], world["tokenizer"], config, 0, 20)
    assert any(len(r["original_shard_ids"]) > 1 for r in results), "no multi-doc bin in the stream"
    assert all(r["matched"] for r in results)


def test_random_mode_offset_shift_formula():
    result = replay_random_mode(start=5)
    assert result["shifted_start"] == 14
    assert result["ledger_bound"] is False


def test_fork_mode_offset_shift_and_branch_change():
    result = replay_fork_mode(checkpoint_offset=20, original_run_branch_id="run-a")
    assert result["new_start_offset"] == 37
    assert result["run_branch_id"] == "run-b"
