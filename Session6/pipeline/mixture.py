"""Mixture schedule: curriculum stage bands, lane profiles, protected floors.

Stage bands and profile percentages mined from the course's mixture-timeline
-compiler widget (`Session6/reference/mined-numbers.md`, widget_7). Protected
lanes (indic/reasoning/agentic) and their combined 8% floor are reused from
Session5's protected-floor decision (`Session5/submission.md` section 6),
split across the three lanes in proportion to their profile weight.
"""

STAGE_BANDS = [
    ("foundation", 0.0, 0.55),
    ("skill_build", 0.55, 0.85),
    ("anneal", 0.85, 1.0),
]

LANES = ["general", "code", "indic", "reasoning", "agentic"]

LANE_PROFILES = {
    "balanced": {"general": 45, "code": 20, "indic": 12, "reasoning": 15, "agentic": 8},
    "code_heavy": {"general": 34, "code": 34, "indic": 10, "reasoning": 14, "agentic": 8},
    "indic_heavy": {"general": 38, "code": 18, "indic": 22, "reasoning": 14, "agentic": 8},
    "anneal": {"general": 36, "code": 21, "indic": 14, "reasoning": 19, "agentic": 10},
}

STAGE_PROFILE = {"foundation": "balanced", "skill_build": "code_heavy", "anneal": "anneal"}

# 8% combined protected floor (Session5 section 6), split by profile weight (12/15/8 of 35).
PROTECTED_FLOOR_PCT = {"indic": 3.0, "reasoning": 3.0, "agentic": 2.0}


def stage_for_fraction(fraction):
    fraction = min(max(fraction, 0.0), 0.999999)
    for name, lo, hi in STAGE_BANDS:
        if lo <= fraction < hi:
            return name
    return STAGE_BANDS[-1][0]


def lane_shares(stage):
    return LANE_PROFILES[STAGE_PROFILE[stage]]


def planned_lane_shares():
    """The mixture the whole run is supposed to realize: each stage's profile
    weighted by how much of the run that stage band covers. Comparing actual
    consumption against any single stage's profile would be wrong - the run
    spans all three bands."""
    planned = {lane: 0.0 for lane in LANES}
    for stage_name, lo, hi in STAGE_BANDS:
        width = hi - lo
        for lane, share in lane_shares(stage_name).items():
            planned[lane] += width * share
    return {lane: round(share, 2) for lane, share in planned.items()}


def feasibility_check(token_budget, supply_by_lane, opus_reject_rate=0.18):
    """supply_by_lane: real measured shard token counts per lane (not fictional demo supply)."""
    warnings = []
    for stage_name, _, _ in STAGE_BANDS:
        for lane, share in lane_shares(stage_name).items():
            need = token_budget * (share / 100) / (1 - opus_reject_rate)
            supply = supply_by_lane.get(lane, 0)
            if supply < need:
                warnings.append({"stage": stage_name, "lane": lane, "need": need, "supply": supply})
    return warnings


class ProtectedFloorTracker:
    """Tracks cumulative realized token share per lane and forces the next
    pick to whichever protected lane has fallen under its floor."""

    def __init__(self, floors=None):
        self.floors = dict(floors if floors is not None else PROTECTED_FLOOR_PCT)
        self.tokens_by_lane = {lane: 0 for lane in LANES}

    def total_tokens(self):
        return sum(self.tokens_by_lane.values())

    def lane_share_pct(self, lane):
        total = self.total_tokens()
        return 100.0 * self.tokens_by_lane[lane] / total if total else 0.0

    def lane_under_floor(self):
        for lane, floor in self.floors.items():
            if self.lane_share_pct(lane) < floor:
                return lane
        return None

    def record(self, lane, token_count):
        self.tokens_by_lane[lane] += token_count


def pick_lane(tracker, stage, rng):
    forced = tracker.lane_under_floor()
    if forced:
        return forced
    shares = lane_shares(stage)
    lanes, weights = zip(*shares.items())
    return rng.choices(lanes, weights=weights, k=1)[0]
