"""Scans the artifacts a run actually generated to build the evidence
bundle. Every row's PASS/FAIL is derived by reading files under
`artifacts_dir` - nothing here is a hardcoded literal.
"""
import hashlib
import json
import os

from pipeline.mixture import PROTECTED_FLOOR_PCT, planned_lane_shares
from pipeline.throughput import recompute_from_inputs

# Lane shares are sampled per step, so realized shares scatter around the plan.
# Wide enough for multinomial noise over a toy run, tight enough to fail if the
# stream collapses onto one lane (the old OPUS scoring drove general to +41).
MAX_LANE_DRIFT_PCT = 8.0


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


def _sample_violations(sample, pad_id):
    """Re-derives every batch-correctness invariant from the recorded arrays."""
    tokens, positions = sample["tokens"], sample["position_ids"]
    segments, mask = sample["segment_ids"], sample["loss_mask"]
    bad = []

    if len({len(tokens), len(positions), len(segments), len(mask)}) != 1:
        bad.append("ragged_arrays")
    if len(tokens) != sample["seq_len"]:
        bad.append("wrong_seq_len")
    if hashlib.sha256(bytes(mask)).hexdigest() != sample["loss_mask_hash"]:
        bad.append("loss_mask_hash_mismatch")
    if any(m and t == pad_id for t, m in zip(tokens, mask)):
        bad.append("padding_is_loss_bearing")
    if segments != sorted(segments):
        bad.append("segment_ids_not_monotonic")

    # Positions must restart at 0 for each document and step by 1 inside it.
    # The trailing pad segment (index == number of documents) is exempt.
    pad_segment = len(sample["shard_ids"])
    expected, prev_segment = 0, None
    for seg, pos in zip(segments, positions):
        if seg >= pad_segment:
            continue
        if seg != prev_segment:
            expected, prev_segment = 0, seg
        if pos != expected:
            bad.append("position_ids_not_reset_per_document")
            break
        expected += 1

    # Every span must be as long as the token run it claims to describe.
    span_lengths = [int(e) - int(s) for _, s, e in
                    (span.rsplit(":", 2) for span in sample["token_span_ids"])]
    run_lengths = [sum(1 for seg in segments if seg == i) for i in range(pad_segment)]
    if span_lengths != run_lengths:
        bad.append("token_span_length_mismatch")
    return bad


def _check_packed_batches(packed, consumption, perf):
    samples = packed["samples"]
    pad_id = packed["pad_id"]
    if not samples:
        return False, "no packed samples recorded"

    violations = {}
    for sample in samples:
        bad = _sample_violations(sample, pad_id)
        if bad:
            violations[sample["ledger_offset"]] = bad

    # The report must describe the same batches the ledger committed.
    committed = {e["ledger_offset"]: e for e in consumption if e["event"] == "batch_committed"}
    mismatched = [s["ledger_offset"] for s in samples
                  if s["ledger_offset"] in committed
                  and (committed[s["ledger_offset"]]["token_span_ids"] != s["token_span_ids"]
                       or committed[s["ledger_offset"]]["loss_mask_hash"] != s["loss_mask_hash"])]

    # And the reported utilization must be recomputable from those same spans.
    used = sum(sum(int(e) - int(s) for _, s, e in
                   (span.rsplit(":", 2) for span in sample["token_span_ids"]))
               for sample in samples)
    capacity = sum(sample["seq_len"] for sample in samples)
    recomputed = round(100.0 * used / capacity, 2) if capacity else 0.0
    util_ok = abs(recomputed - perf.get("packing_utilization_pct", -1)) < 0.01

    multi_doc = sum(1 for s in samples if len(s["shard_ids"]) > 1)
    passed = not violations and not mismatched and util_ok and multi_doc > 0
    detail = (f"{len(samples)} packed samples re-verified (masks, segment ids, position resets, "
              f"span lengths); {multi_doc} multi-document; utilization recomputed "
              f"{recomputed}% vs reported {perf.get('packing_utilization_pct')}%")
    if violations:
        detail += f"; VIOLATIONS {violations}"
    if mismatched:
        detail += f"; ledger mismatch at {mismatched}"
    return passed, detail


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
    blocked_manifests = [m for m in manifests if m["eval_overlap_status"] != "clear"]
    firewall_ok = (bool(blocked_manifests)
                   and all(m["shard_id"] not in trained_shard_ids for m in blocked_manifests)
                   # ...and the admission gate must refuse them, not merely the
                   # training loop happening never to select them.
                   and all(m["admission"] == "blocked" for m in blocked_manifests))
    rows.append(_row("Evaluation firewall", "eval_firewall", firewall_ok, artifacts_dir,
                      consumption_path,
                      f"{len(blocked_manifests)} firewall-blocked shards, all admission=blocked, "
                      f"none present in trained shard ids"))

    # Every one of these fields used to be a declared constant or a default
    # argument. The row fails if the cleaning pass stops finding anything,
    # which is what "measured" degrading back to "asserted" would look like.
    contaminated = [m for m in manifests if m["benchmark_overlap_pct"] > 0]
    canaries = [m for m in manifests if m["canary_match"]]
    pii = [m for m in manifests if m["pii_screen_status"] == "masked"]
    dupes = [m for m in manifests if m["dedup_status"].startswith("near_duplicate_of:")]
    varied = len({m["dedup_status"] for m in manifests}) > 1 and \
        len({m["pii_screen_status"] for m in manifests}) > 1
    cleaning_ok = (bool(contaminated) and bool(canaries) and bool(pii) and bool(dupes) and varied
                   and all(m["admission"] == "blocked" for m in dupes)
                   and all(sum(m["pii_counts"].values()) > 0 for m in pii))
    rows.append(_row("Data cleaning", "shards_manifests_tokenizer", cleaning_ok, artifacts_dir,
                      manifests_path,
                      f"measured: {len(contaminated)} eval-overlapping, {len(canaries)} canary, "
                      f"{len(pii)} PII-masked ({sum(sum(m['pii_counts'].values()) for m in pii)} "
                      f"identifiers), {len(dupes)} near-duplicate (all blocked)"))

    perf_path = os.path.join(artifacts_dir, "performance.json")
    perf = _load_json(perf_path)
    packed_path = os.path.join(artifacts_dir, "packed_batches.json")
    packed = _load_json(packed_path)
    packing_ok, packing_detail = _check_packed_batches(packed, consumption, perf)
    rows.append(_row("Packing correctness", "packing_masks_batches", packing_ok, artifacts_dir,
                      packed_path, packing_detail))

    learning_path = os.path.join(artifacts_dir, "ledgers", "learning.jsonl")
    learning = _load_jsonl(learning_path)
    tokens_by_lane = {}
    for e in learning:
        tokens_by_lane[e["lane"]] = tokens_by_lane.get(e["lane"], 0) + e["tokens_consumed"]
    total_tokens = sum(tokens_by_lane.values()) or 1
    actual = {lane: round(100.0 * tok / total_tokens, 2) for lane, tok in tokens_by_lane.items()}
    planned = planned_lane_shares()

    floor_ok = all(actual.get(lane, 0.0) >= floor - 0.5 for lane, floor in PROTECTED_FLOOR_PCT.items())
    # Every lane in the plan must actually appear, and realized shares must
    # track the planned mixture rather than collapsing onto one lane.
    drift = {lane: round(actual.get(lane, 0.0) - share, 2) for lane, share in planned.items()}
    all_lanes_present = all(actual.get(lane, 0.0) > 0 for lane, share in planned.items() if share > 0)
    drift_ok = all(abs(d) <= MAX_LANE_DRIFT_PCT for d in drift.values())
    mixture_ok = floor_ok and all_lanes_present and drift_ok

    rows.append(_row("Mixture compliance", "mixture_floors_opus", mixture_ok, artifacts_dir,
                      learning_path,
                      f"planned {planned} vs actual {actual}; drift {drift} "
                      f"(max allowed +/-{MAX_LANE_DRIFT_PCT}); floors {dict(PROTECTED_FLOOR_PCT)} "
                      f"{'met' if floor_ok else 'BREACHED'}"))

    opus_path = os.path.join(artifacts_dir, "ledgers", "opus_decisions.jsonl")
    opus_decisions = _load_jsonl(opus_path)
    committed_decision_ids = {e["opus_decision_id"] for e in consumption if e["event"] == "batch_committed"}
    logged_decision_ids = {d["opus_decision_id"] for d in opus_decisions}
    kinds = {d["decision"] for d in opus_decisions}
    overrides = sum(1 for d in opus_decisions if d["protected_floor_override"])
    # An audit trail that only records the accepted candidates is not an audit
    # trail - rejection, deferral and floor override must be in it too. The
    # `protected` *decision* is the narrower case where the floor also rescued
    # a candidate the proxy score would have dropped; under a healthy mixture
    # the floor mostly binds without needing to rescue anything, so the
    # override count is what has to be non-zero here.
    opus_ok = (bool(opus_decisions) and committed_decision_ids.issubset(logged_decision_ids)
               and {"accepted", "rejected", "deferred"}.issubset(kinds)
               and overrides > 0
               and all(d["rejection_reason"] for d in opus_decisions if d["decision"] == "rejected"))
    counts = {k: sum(1 for d in opus_decisions if d["decision"] == k) for k in sorted(kinds)}
    rows.append(_row("OPUS audit trail", "mixture_floors_opus", opus_ok, artifacts_dir,
                      opus_path, f"{len(opus_decisions)} candidate decisions logged: {counts}; "
                                 f"{overrides} protected-floor overrides"))

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
