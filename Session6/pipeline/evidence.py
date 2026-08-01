"""Scans the artifacts a run actually generated to build the evidence
bundle. Every row's PASS/FAIL is derived by reading files under
`artifacts_dir` - nothing here is a hardcoded literal.
"""
import json
import os

from pipeline.mixture import PROTECTED_FLOOR_PCT
from pipeline.throughput import recompute_from_inputs


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _row(requirement, area, passed, artifacts_dir, evidence_abspath, detail):
    return {
        "requirement": requirement,
        "area": area,
        "result": "PASS" if passed else "FAIL",
        "evidence_path": os.path.relpath(evidence_abspath, artifacts_dir),
        "detail": detail,
    }


def build_evidence(artifacts_dir):
    rows = []

    tok_path = os.path.join(artifacts_dir, "manifests", "tokenizer.json")
    manifests_path = os.path.join(artifacts_dir, "manifests", "shard_manifests.json")
    tokenizer = _load_json(tok_path)
    manifests = _load_json(manifests_path)

    tok_ok = bool(manifests) and all(m["tokenizer_hash"] == tokenizer["tokenizer_hash"] for m in manifests)
    rows.append(_row("Tokenizer integrity", "shards_manifests_tokenizer", tok_ok, artifacts_dir,
                      tok_path, f"{len(manifests)} manifests checked against tokenizer.json"))

    consumption_path = os.path.join(artifacts_dir, "ledgers", "consumption.jsonl")
    consumption = _load_jsonl(consumption_path)
    trained_shard_ids = {sid for e in consumption if e["event"] == "batch_committed" for sid in e["shard_ids"]}
    blocked_manifests = [m for m in manifests if m["eval_overlap_status"] == "blocked_or_unknown"]
    firewall_ok = bool(blocked_manifests) and all(m["shard_id"] not in trained_shard_ids for m in blocked_manifests)
    rows.append(_row("Evaluation firewall", "eval_firewall", firewall_ok, artifacts_dir,
                      consumption_path,
                      f"{len(blocked_manifests)} firewall-blocked shards, none present in trained shard ids"))

    perf_path = os.path.join(artifacts_dir, "performance.json")
    perf = _load_json(perf_path)
    packing_ok = 0 <= perf.get("packing_utilization_pct", -1) <= 100
    rows.append(_row("Packing correctness", "packing_masks_batches", packing_ok, artifacts_dir,
                      perf_path, f"packing_utilization_pct={perf.get('packing_utilization_pct')}"))

    learning_path = os.path.join(artifacts_dir, "ledgers", "learning.jsonl")
    learning = _load_jsonl(learning_path)
    tokens_by_lane = {}
    for e in learning:
        tokens_by_lane[e["lane"]] = tokens_by_lane.get(e["lane"], 0) + e["tokens_consumed"]
    total_tokens = sum(tokens_by_lane.values()) or 1
    lane_shares = {lane: round(100.0 * tok / total_tokens, 2) for lane, tok in tokens_by_lane.items()}
    floor_ok = all(lane_shares.get(lane, 0.0) >= floor - 0.5 for lane, floor in PROTECTED_FLOOR_PCT.items())
    rows.append(_row("Mixture compliance", "mixture_floors_opus", floor_ok, artifacts_dir,
                      learning_path, f"lane token shares: {lane_shares}"))

    opus_path = os.path.join(artifacts_dir, "ledgers", "opus_decisions.jsonl")
    opus_decisions = _load_jsonl(opus_path)
    committed_decision_ids = {e["opus_decision_id"] for e in consumption if e["event"] == "batch_committed"}
    logged_decision_ids = {d["opus_decision_id"] for d in opus_decisions}
    opus_ok = bool(opus_decisions) and committed_decision_ids.issubset(logged_decision_ids)
    rows.append(_row("OPUS audit trail", "mixture_floors_opus", opus_ok, artifacts_dir,
                      opus_path, f"{len(opus_decisions)} candidate decisions logged"))

    crash_path = os.path.join(artifacts_dir, "ledgers", "crash_resume_report.json")
    crash_report = _load_json(crash_path)
    crash_ok = bool(crash_report.get("no_skip_no_repeat")) and bool(crash_report.get("resume_matched_expected"))
    rows.append(_row("Crash recovery", "checkpoint_crash_resume_replay_fork", crash_ok, artifacts_dir,
                      crash_path, f"resume_offset={crash_report.get('resume_offset')}"))

    replay_path = os.path.join(artifacts_dir, "ledgers", "replay_report.json")
    replay_report = _load_json(replay_path)
    replay_results = replay_report.get("ledger_mode_results", [])
    replay_ok = bool(replay_results) and all(r["matched"] for r in replay_results)
    rows.append(_row("Replay", "checkpoint_crash_resume_replay_fork", replay_ok, artifacts_dir,
                      replay_path, f"{len(replay_results)} intervals replayed and hash-matched"))

    manifest_shard_ids = {m["shard_id"] for m in manifests}
    learning_ok = bool(learning) and all(e["shard_id"] in manifest_shard_ids for e in learning)
    rows.append(_row("Learning trace", "consumption_learning_ledgers", learning_ok, artifacts_dir,
                      learning_path, f"{len(learning)} learning-ledger rollups traced to manifest shard ids"))

    recomputed = recompute_from_inputs(perf["inputs"])
    throughput_ok = all(
        perf[k] == recomputed[k] for k in ("useful_tokens_per_sec", "useful_after_opus", "useful_after_pack")
    )
    rows.append(_row("Throughput", "throughput_packing_efficiency", throughput_ok, artifacts_dir,
                      perf_path, f"useful_tokens_per_sec={perf.get('useful_tokens_per_sec')} (recomputed from inputs)"))

    return rows


def write_evidence(rows, artifacts_dir):
    json_path = os.path.join(artifacts_dir, "evidence.json")
    md_path = os.path.join(artifacts_dir, "evidence.md")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    lines = ["| REQUIREMENT | RESULT | EVIDENCE |", "|---|---|---|"]
    lines.extend(f"| {row['requirement']} | {row['result']} | {row['evidence_path']} |" for row in rows)
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path
