import json
import os

from pipeline.crash_resume import (simulate_crash_and_resume, verify_no_skip_no_repeat,
                                    verify_resume_matched_expected)
from pipeline.evidence import build_evidence, write_evidence
from pipeline.ledger import JsonlLedger
from pipeline.packing import packing_utilization
from pipeline.replay import replay_ledger_mode
from pipeline.throughput import compute_performance
from pipeline.train_loop import Config, build_world, materialize_plan, run_plan


def _build_full_artifacts(tmp_path):
    artifacts_dir = str(tmp_path / "submission_artifacts")
    os.makedirs(os.path.join(artifacts_dir, "manifests"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "ledgers"), exist_ok=True)
    checkpoint_dir = os.path.join(artifacts_dir, "checkpoints")

    config = Config(total_steps=200, ckpt_interval=10)
    world = build_world(config)
    stats = {}
    plan = materialize_plan(world, config, stats=stats)
    assert len(plan) > 20

    with open(os.path.join(artifacts_dir, "manifests", "tokenizer.json"), "w") as f:
        json.dump(world["tokenizer"].to_dict(), f)
    all_manifests = world["training_manifests"] + world["eval_manifests"]
    with open(os.path.join(artifacts_dir, "manifests", "shard_manifests.json"), "w") as f:
        json.dump(all_manifests, f)

    consumption = JsonlLedger(os.path.join(artifacts_dir, "ledgers", "consumption.jsonl"))
    learning = JsonlLedger(os.path.join(artifacts_dir, "ledgers", "learning.jsonl"))
    opus_ledger = JsonlLedger(os.path.join(artifacts_dir, "ledgers", "opus_decisions.jsonl"))

    crash_result = simulate_crash_and_resume(plan, config, consumption, learning, checkpoint_dir, "run-a")
    # opus decisions weren't logged during simulate_crash_and_resume (it calls run_plan internally
    # without opus_ledger); replay a second, independent pass over the plan just to log them.
    for entry in plan:
        opus_ledger.append(entry["opus_decision"])

    crash_report = {
        "no_skip_no_repeat": verify_no_skip_no_repeat(consumption, len(plan)),
        "resume_matched_expected": verify_resume_matched_expected(consumption, plan, crash_result["resume_offset"]),
        "resume_offset": crash_result["resume_offset"],
    }
    with open(os.path.join(artifacts_dir, "ledgers", "crash_resume_report.json"), "w") as f:
        json.dump(crash_report, f)

    replay_results = replay_ledger_mode(consumption, world["training_shards"], world["tokenizer"], config, 0, 10)
    with open(os.path.join(artifacts_dir, "ledgers", "replay_report.json"), "w") as f:
        json.dump({"ledger_mode_results": replay_results}, f)

    bins_by_lane = {}
    from pipeline.packing import pack_lane
    for lane, shards in world["admitted_by_lane"].items():
        bins_by_lane.setdefault(lane, []).extend(
            pack_lane(lane, shards, seq_len=config.seq_len, eos_id=world["tokenizer"].eos_id))
    all_bins = [b for bins in bins_by_lane.values() for b in bins]
    pack_pct = packing_utilization(all_bins, seq_len=config.seq_len)

    decisions = stats["decision_counts"]
    total = sum(decisions.values()) or 1
    reject_pct = 100.0 * (decisions.get("rejected", 0) + decisions.get("deferred", 0)) / total

    perf = compute_performance(config, pack_pct=pack_pct, reject_pct=reject_pct)
    with open(os.path.join(artifacts_dir, "performance.json"), "w") as f:
        json.dump(perf, f)

    return artifacts_dir


def test_all_evidence_paths_exist(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)
    rows = build_evidence(artifacts_dir)
    for row in rows:
        full_path = os.path.join(artifacts_dir, row["evidence_path"])
        assert os.path.exists(full_path), f"missing evidence path for {row['requirement']}"


def test_evidence_bundle_writes_json_and_md(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)
    rows = build_evidence(artifacts_dir)
    json_path, md_path = write_evidence(rows, artifacts_dir)
    assert os.path.exists(json_path)
    assert os.path.exists(md_path)
    with open(json_path) as f:
        assert len(json.load(f)) == len(rows)


def test_all_rows_pass_on_a_healthy_run(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)
    rows = build_evidence(artifacts_dir)
    failing = [r for r in rows if r["result"] == "FAIL"]
    assert not failing, failing


def test_evidence_reflects_induced_failure(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)
    # Deliberately break the crash-recovery invariant to prove evidence isn't hardcoded.
    report_path = os.path.join(artifacts_dir, "ledgers", "crash_resume_report.json")
    with open(report_path, "w") as f:
        json.dump({"no_skip_no_repeat": False, "resume_matched_expected": False, "resume_offset": 0}, f)
    rows = build_evidence(artifacts_dir)
    crash_row = next(r for r in rows if r["requirement"] == "Crash recovery")
    assert crash_row["result"] == "FAIL"
