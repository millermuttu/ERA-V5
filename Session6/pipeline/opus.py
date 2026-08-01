"""OPUS admission decisions: accepted / rejected / deferred / protected.

Decision taxonomy and rejection-reason enum mined from the course's
opus-audit-board widget (`Session6/reference/mined-numbers.md`, widget_10).
Scoring is a deterministic hash of `candidate_batch_id` blended with the
lane's mixture-stage target weight, so replays and resumes always agree -
no seeded `random` state to lose across a crash boundary.
"""
import hashlib

from pipeline.firewall import check as firewall_check

PROXY_VERSION = "session6-toy-proxy-v2"
ACCEPT_THRESHOLD = 0.28
# Annealing reserve: hold part of these lanes back during the earliest stage
# only. Deferring them for every non-anneal stage would starve them across 85%
# of the run and leave the protected floor as their sole route into training.
DEFER_TO_ANNEAL_LANES = {"indic", "reasoning"}
DEFER_STAGES = {"foundation"}

REJECTION_REASONS = [
    "below_proxy_threshold", "lane_quota_full", "duplicate_update_direction",
    "stage_mismatch", "eval_firewall_overlap", "deferred_for_anneal",
]


def jitter(candidate_batch_id):
    digest = hashlib.sha256(candidate_batch_id.encode()).hexdigest()
    return int(digest, 16) / (2 ** 256)


def score_candidate(candidate_batch_id, lane=None, stage=None):
    """Quality proxy for a candidate batch - deliberately independent of the
    lane's mixture weight.

    Blending the lane weight in here double-counts the mixture: `pick_lane`
    has already sampled the lane by its profile share, so scoring by that
    same share again re-filters an already-correct stream and collapses it
    onto whichever lane is heaviest. (With the old `0.6*weight + 0.4*jitter`
    against a 0.6 threshold, a lane could only ever be accepted if its share
    exceeded 1/3, so every lane but `general` depended entirely on the
    protected-floor override to enter training at all.)

    `lane`/`stage` are accepted and ignored so callers read naturally.
    """
    return round(jitter(candidate_batch_id), 4)


def decide(candidate_batch_id, shards, lane, stage, model_age, lane_under_floor=False):
    """shards: shard dicts backing this candidate batch.

    Order matters: a protected-floor pick must resolve to accepted/protected
    (never deferred/rejected) so the mixture's floor guarantee actually
    holds - only the eval firewall outranks the floor override.
    """
    opus_decision_id = f"opus-{candidate_batch_id}"

    if any(not firewall_check(s)["passed"] for s in shards):
        return _record(opus_decision_id, candidate_batch_id, model_age, lane, stage,
                        0.0, "rejected", "eval_firewall_overlap", shards)

    score = score_candidate(candidate_batch_id, lane, stage)

    if score >= ACCEPT_THRESHOLD:
        # The floor still overrode normal mixture sampling to put this lane
        # here, even though the proxy score would have admitted it anyway.
        # Recording that only on the rescue path under-reports how often the
        # floor actually steered the stream.
        return _record(opus_decision_id, candidate_batch_id, model_age, lane, stage,
                        score, "accepted", None, shards,
                        protected_floor_override=lane_under_floor)

    if lane_under_floor:
        return _record(opus_decision_id, candidate_batch_id, model_age, lane, stage,
                        score, "protected", None, shards, protected_floor_override=True)

    if stage in DEFER_STAGES and lane in DEFER_TO_ANNEAL_LANES:
        return _record(opus_decision_id, candidate_batch_id, model_age, lane, stage,
                        score, "deferred", "deferred_for_anneal", shards)

    return _record(opus_decision_id, candidate_batch_id, model_age, lane, stage,
                    score, "rejected", "below_proxy_threshold", shards)


def _record(opus_decision_id, candidate_batch_id, model_age, lane, stage, score, decision,
            rejection_reason, shards, protected_floor_override=False):
    return {
        "opus_decision_id": opus_decision_id,
        "candidate_batch_id": candidate_batch_id,
        "model_age": model_age,
        "proxy_version": PROXY_VERSION,
        "lane": lane,
        "stage": stage,
        "score": score,
        "decision": decision,
        "rejection_reason": rejection_reason,
        "protected_floor_override": protected_floor_override,
        "shard_ids": [s["shard_id"] for s in shards],
        "effective_token_estimate": sum(s["token_count"] for s in shards),
    }
