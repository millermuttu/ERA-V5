"""Shard manifest schema and admission-gate scoring.

Score formula and thresholds mined from the course's shard-manifest-builder
widget (`Session6/reference/mined-numbers.md`): score is 78% required-field
completeness + 22% license score, capped at 64 if any hard-required field
is missing or the license is unsafe; admitted only if score > 86.
"""
import hashlib

from pipeline.firewall import check as firewall_check

HARD_REQUIRED_FIELDS = ["tokenizer_hash", "cleaning_pipeline_hash", "eval_overlap_status"]
ADMIT_THRESHOLD = 86
HARD_BLOCK_CAP = 64
UNSAFE_LICENSE = "unsafe"
_LICENSE_SCORE = {"safe": 100, "review": 60, "unsafe": 0}

CLEANING_PIPELINE_VERSION = "session6-toy-clean-v1"


def cleaning_pipeline_hash():
    return hashlib.sha256(CLEANING_PIPELINE_VERSION.encode()).hexdigest()


def is_hard_blocked(manifest):
    missing = [f for f in HARD_REQUIRED_FIELDS if not manifest.get(f)]
    return bool(missing) or manifest.get("license_tier") == UNSAFE_LICENSE


def score_manifest(manifest):
    present = sum(1 for f in HARD_REQUIRED_FIELDS if manifest.get(f)) / len(HARD_REQUIRED_FIELDS)
    license_score = _LICENSE_SCORE.get(manifest.get("license_tier"), 0)
    score = 78 * present + 22 * (license_score / 100)
    if is_hard_blocked(manifest):
        score = min(score, HARD_BLOCK_CAP)
    return round(score, 2)


def decide_admission(manifest):
    if manifest.get("license_tier") == UNSAFE_LICENSE:
        return "blocked"
    if is_hard_blocked(manifest):
        return "held_for_review"
    if manifest["admission_score"] > ADMIT_THRESHOLD:
        return "admitted"
    return "held_for_review"


def build_manifest(shard, license_tier="safe", dedup_status="passed", pii_screen_status="screened"):
    firewall_result = firewall_check(shard)
    eval_overlap_status = "clear" if firewall_result["passed"] else "blocked_or_unknown"
    manifest = {
        "shard_id": shard["shard_id"],
        "capability_lane": shard["lane"],
        "doc_ids": shard["doc_ids"],
        "token_count": shard["token_count"],
        "tokenizer_hash": shard["tokenizer_hash"],
        "content_hash": shard["content_hash"],
        "cleaning_pipeline_hash": cleaning_pipeline_hash(),
        "dedup_status": dedup_status,
        "pii_screen_status": pii_screen_status,
        "eval_overlap_status": eval_overlap_status,
        "license_tier": license_tier,
        "parent_manifest_ids": [],
    }
    manifest["admission_score"] = score_manifest(manifest)
    manifest["admission"] = decide_admission(manifest)
    return manifest


def build_manifests(shards, **kwargs):
    return [build_manifest(shard, **kwargs) for shard in shards]
