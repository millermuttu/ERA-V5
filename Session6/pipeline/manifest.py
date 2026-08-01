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
EVAL_CLEAR = "clear"
EVAL_BLOCKED = "blocked_or_unknown"
_LICENSE_SCORE = {"safe": 100, "review": 60, "unsafe": 0}

CLEANING_PIPELINE_VERSION = "session6-toy-clean-v1"


def cleaning_pipeline_hash():
    return hashlib.sha256(CLEANING_PIPELINE_VERSION.encode()).hexdigest()


def firewall_tripped(manifest):
    """A manifest whose shard failed the eval firewall can never be admitted -
    the presence of the `eval_overlap_status` field is not enough, its value
    has to be `clear`."""
    return manifest.get("eval_overlap_status") != EVAL_CLEAR


def is_hard_blocked(manifest):
    missing = [f for f in HARD_REQUIRED_FIELDS if not manifest.get(f)]
    return (bool(missing) or manifest.get("license_tier") == UNSAFE_LICENSE
            or firewall_tripped(manifest))


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
    if firewall_tripped(manifest):
        return "blocked"
    # A near-duplicate of an already-admitted shard adds no new tokens and
    # would inflate that content's effective epoch count.
    if str(manifest.get("dedup_status", "")).startswith("near_duplicate_of:"):
        return "blocked"
    if is_hard_blocked(manifest):
        return "held_for_review"
    if manifest["admission_score"] > ADMIT_THRESHOLD:
        return "admitted"
    return "held_for_review"


def build_manifest(shard, license_tier="safe", duplicate_of=None):
    """`dedup_status` and `pii_screen_status` are derived from what the
    cleaning pass actually found on this shard - they used to be default
    arguments asserting "passed"/"screened" with no check behind them."""
    firewall_result = firewall_check(shard)
    eval_overlap_status = EVAL_CLEAR if firewall_result["passed"] else EVAL_BLOCKED
    dedup_status = f"near_duplicate_of:{duplicate_of}" if duplicate_of else "unique"
    pii_counts = shard.get("pii_counts", {})
    pii_screen_status = ("masked" if shard.get("pii_found") else "clean")
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
        "pii_counts": pii_counts,
        "eval_overlap_status": eval_overlap_status,
        "eval_firewall_reasons": firewall_result["reasons"],
        # Measured by pipeline/cleaning.py, recorded here so the gate decision
        # is auditable from the manifest alone rather than only reproducible.
        "benchmark_overlap_pct": shard.get("benchmark_overlap_pct", 0.0),
        "canary_match": shard.get("canary_match", False),
        "never_train": shard.get("never_train", False),
        "benchmark_derived": shard.get("benchmark_derived", False),
        "license_tier": license_tier,
        "parent_manifest_ids": [],
    }
    manifest["admission_score"] = score_manifest(manifest)
    manifest["admission"] = decide_admission(manifest)
    return manifest


def build_manifests(shards, duplicates=None, **kwargs):
    """duplicates: {shard_id: kept_shard_id} from cleaning.near_duplicates."""
    duplicates = duplicates or {}
    return [build_manifest(shard, duplicate_of=duplicates.get(shard["shard_id"]), **kwargs)
            for shard in shards]
