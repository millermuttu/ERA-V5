"""Audit: reconstruct which shards and OPUS decisions produced the
committed batches behind a given ledger offset range or checkpoint."""


def audit_range(ledger, start, end):
    events = [e for e in ledger.read_all()
              if e["event"] == "batch_committed" and start <= e["ledger_offset"] < end]
    return {
        "range": [start, end],
        "batch_count": len(events),
        "shard_ids": sorted({sid for e in events for sid in e["shard_ids"]}),
        "opus_decision_ids": sorted({e["opus_decision_id"] for e in events}),
        "mixture_lanes": sorted({e["mixture_lane"] for e in events}),
        "curriculum_stages": sorted({e["curriculum_stage"] for e in events}),
    }


def audit_behind_checkpoint(ledger, checkpoint):
    return audit_range(ledger, 0, checkpoint["ledger_offset"] + 1)
