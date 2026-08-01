from pipeline.ledger import JsonlLedger, batch_committed_event, checkpoint_bound_event


def _sample(shard_ids=("s1",)):
    return {"shard_ids": list(shard_ids), "tokens": [1, 2, 3], "loss_mask_hash": "hash1", "policy": "pad_only"}


def _opus_decision(decision_id="opus-1"):
    return {"opus_decision_id": decision_id}


def test_ledger_offsets_strictly_increase(tmp_path):
    ledger = JsonlLedger(str(tmp_path / "consumption.jsonl"))
    for offset in range(5):
        event = batch_committed_event(offset, "run-a", offset, _sample(), _opus_decision(),
                                       "general", "foundation")
        ledger.append(event)
    events = ledger.read_all()
    offsets = [e["ledger_offset"] for e in events]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


def test_ledger_is_append_only(tmp_path):
    ledger = JsonlLedger(str(tmp_path / "consumption.jsonl"))
    ledger.append(batch_committed_event(0, "run-a", 0, _sample(), _opus_decision(), "general", "foundation"))
    assert len(ledger.read_all()) == 1
    ledger.append(batch_committed_event(1, "run-a", 1, _sample(), _opus_decision(), "general", "foundation"))
    assert len(ledger.read_all()) == 2


def test_checkpoint_bound_references_valid_offset(tmp_path):
    ledger = JsonlLedger(str(tmp_path / "consumption.jsonl"))
    for offset in range(3):
        ledger.append(batch_committed_event(offset, "run-a", offset, _sample(), _opus_decision(),
                                             "general", "foundation"))
    ckpt_event = checkpoint_bound_event("ckpt-1", 2, "run-a", "modelhash", "opthash", "rngstate")
    ledger.append(ckpt_event)

    events = ledger.read_all()
    committed_offsets = {e["ledger_offset"] for e in events if e["event"] == "batch_committed"}
    assert ckpt_event["ledger_offset"] in committed_offsets
