"""OPUS admission decisions: accepted / rejected / deferred / protected.

Decision taxonomy and rejection-reason enum mined from the course's
opus-audit-board widget (`Session6/reference/mined-numbers.md`, widget_10).
Scoring is a deterministic hash of `candidate_batch_id` blended with the
lane's mixture-stage target weight, so replays and resumes always agree -
no seeded `random` state to lose across a crash boundary.
"""
import hashlib

from pipeline.firewall import check as firewall_check
from pipeline.mixture import lane_shares

PROXY_VERSION = "session6-toy-proxy-v1"
ACCEPT_THRESHOLD = 0.6
DEFER_TO_ANNEAL_LANES = {"indic", "reasoning"}

REJECTION_REASONS = [
    "below_proxy_threshold", "lane_quota_full", "duplicate_update_direction",
    "stage_mismatch", "eval_firewall_overlap", "deferred_for_anneal",
]


def jitter(candidate_batch_id):
    digest = hashlib.sha256(candidate_batch_id.encode()).hexdigest()
    return int(digest, 16) / (2 ** 256)


def score_candidate(candidate_batch_id, lane, stage):
    weight = lane_shares(stage).get(lane, 0) / 100
    return round(0.6 * weight + 0.4 * jitter(candidate_batch_id), 4)


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
        return _record(opus_decision_id, candidate_batch_id, model_age, lane, stage,
                        score, "accepted", None, shards)

    if lane_under_floor:
        return _record(opus_decision_id, candidate_batch_id, model_age, lane, stage,
                        score, "protected", None, shards, protected_floor_override=True)

    if stage != "anneal" and lane in DEFER_TO_ANNEAL_LANES:
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
