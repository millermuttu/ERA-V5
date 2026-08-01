import random

from pipeline.mixture import (PROTECTED_FLOOR_PCT, STAGE_BANDS, ProtectedFloorTracker,
                               lane_shares, pick_lane, stage_for_fraction)


def test_stage_bands_partition():
    assert stage_for_fraction(0.0) == "foundation"
    assert stage_for_fraction(0.54) == "foundation"
    assert stage_for_fraction(0.55) == "skill_build"
    assert stage_for_fraction(0.84) == "skill_build"
    assert stage_for_fraction(0.85) == "anneal"
    assert stage_for_fraction(0.999) == "anneal"


def test_profiles_sum_to_100():
    for stage_name, _, _ in STAGE_BANDS:
        assert sum(lane_shares(stage_name).values()) == 100


def test_protected_floor_holds_over_many_draws():
    rng = random.Random(0)
    tracker = ProtectedFloorTracker()
    for i in range(3000):
        stage = stage_for_fraction(i / 3000)
        lane = pick_lane(tracker, stage, rng)
        tracker.record(lane, 1)
    for lane, floor in PROTECTED_FLOOR_PCT.items():
        assert tracker.lane_share_pct(lane) >= floor - 0.5
