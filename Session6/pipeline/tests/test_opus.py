from pipeline.opus import decide, score_candidate


def _shard(**overrides):
    base = {"shard_id": "general-abc123", "token_count": 100, "never_train": False,
            "benchmark_overlap_pct": 0.0, "canary_match": False, "benchmark_derived": False}
    base.update(overrides)
    return base


def test_deterministic_score_for_fixed_candidate():
    a = score_candidate("cand-42", "general", "foundation")
    b = score_candidate("cand-42", "general", "foundation")
    assert a == b


def test_deterministic_decision_for_fixed_candidate():
    shards = [_shard()]
    d1 = decide("cand-42", shards, "general", "foundation", model_age=10)
    d2 = decide("cand-42", shards, "general", "foundation", model_age=10)
    assert d1["decision"] == d2["decision"]
    assert d1["score"] == d2["score"]


def test_rejected_decision_always_has_reason():
    shards = [_shard()]
    for cand_id in [f"cand-{i}" for i in range(50)]:
        d = decide(cand_id, shards, "general", "foundation", model_age=0)
        if d["decision"] == "rejected":
            assert d["rejection_reason"] is not None


def test_firewall_flagged_candidate_never_gets_protected_override():
    shards = [_shard(never_train=True)]
    d = decide("cand-firewall", shards, "indic", "foundation", model_age=0, lane_under_floor=True)
    assert d["decision"] == "rejected"
    assert d["rejection_reason"] == "eval_firewall_overlap"
    assert d["protected_floor_override"] is False
