"""Evaluation and validation firewall - single canonical gate function.

Called at two sites (manifest admission and batch assembly) so the four
gates live in exactly one place. A shard trips the firewall if it is
flagged never-train, exceeds the benchmark-overlap threshold, matches a
canary string, or is derived from benchmark content - regardless of the
shard's lane protected-floor status.
"""

OVERLAP_THRESHOLD_PCT = 25.0


def check(shard):
    reasons = []
    if shard.get("never_train"):
        reasons.append("never_train")
    if shard.get("benchmark_overlap_pct", 0.0) > OVERLAP_THRESHOLD_PCT:
        reasons.append("benchmark_overlap_exceeded")
    if shard.get("canary_match"):
        reasons.append("canary_match")
    if shard.get("benchmark_derived"):
        reasons.append("benchmark_derived")
    return {"passed": not reasons, "reasons": reasons}
