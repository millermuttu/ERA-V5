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


def test_no_lane_is_structurally_unacceptable():
    """Regression: scoring blended the lane's mixture weight in, so any lane
    with a share under 1/3 could never clear the threshold on merit and
    depended entirely on the protected-floor override to train at all."""
    from pipeline.mixture import LANE_PROFILES, STAGE_PROFILE
    for stage in STAGE_PROFILE:
        for lane in LANE_PROFILES[STAGE_PROFILE[stage]]:
            accepted = [decide(f"cand-{i}", [_shard()], lane, stage, model_age=0)["decision"]
                        for i in range(200)]
            assert "accepted" in accepted, f"{lane} can never be accepted in {stage}"


def test_score_is_independent_of_lane_and_stage():
    """The mixture is applied once, by pick_lane - OPUS must not re-apply it."""
    base = score_candidate("cand-7", "general", "foundation")
    for lane in ("general", "code", "indic", "reasoning", "agentic"):
        for stage in ("foundation", "skill_build", "anneal"):
            assert score_candidate("cand-7", lane, stage) == base


def test_realized_lane_shares_track_the_planned_mixture():
    from pipeline.evidence import MAX_LANE_DRIFT_PCT
    from pipeline.mixture import planned_lane_shares
    from pipeline.train_loop import Config, build_world, materialize_plan

    config = Config()
    plan = materialize_plan(build_world(config), config)
    tokens = {}
    for entry in plan:
        for shard in entry["shards"]:
            tokens[entry["lane"]] = tokens.get(entry["lane"], 0) + shard["token_count"]
    total = sum(tokens.values())
    for lane, planned in planned_lane_shares().items():
        actual = 100.0 * tokens.get(lane, 0) / total
        assert abs(actual - planned) <= MAX_LANE_DRIFT_PCT, \
            f"{lane}: planned {planned}%, actual {actual:.2f}%"


def test_floor_rescues_a_candidate_the_proxy_would_have_dropped():
    """The `protected` decision: a below-threshold candidate that the floor
    keeps anyway. Rare in the shipped run because a healthy mixture keeps
    every lane above its floor, so it is pinned here rather than left to
    chance in the artifacts."""
    from pipeline.opus import ACCEPT_THRESHOLD, score_candidate
    low = next(f"cand-{i}" for i in range(500)
               if score_candidate(f"cand-{i}") < ACCEPT_THRESHOLD)

    dropped = decide(low, [_shard()], "indic", "skill_build", model_age=0, lane_under_floor=False)
    rescued = decide(low, [_shard()], "indic", "skill_build", model_age=0, lane_under_floor=True)

    assert dropped["decision"] == "rejected"
    assert rescued["decision"] == "protected"
    assert rescued["protected_floor_override"] is True


def test_floor_override_is_recorded_even_when_the_proxy_would_accept():
    """The floor overrode lane selection regardless of the proxy's verdict;
    recording that only on the rescue path under-reports it."""
    from pipeline.opus import ACCEPT_THRESHOLD, score_candidate
    high = next(f"cand-{i}" for i in range(500)
                if score_candidate(f"cand-{i}") >= ACCEPT_THRESHOLD)
    d = decide(high, [_shard()], "indic", "skill_build", model_age=0, lane_under_floor=True)
    assert d["decision"] == "accepted"
    assert d["protected_floor_override"] is True


def test_firewall_flagged_candidate_never_gets_protected_override():
    shards = [_shard(never_train=True)]
    d = decide("cand-firewall", shards, "indic", "foundation", model_age=0, lane_under_floor=True)
    assert d["decision"] == "rejected"
    assert d["rejection_reason"] == "eval_firewall_overlap"
    assert d["protected_floor_override"] is False
