import hashlib
import json
import os

from pipeline.crash_resume import (simulate_crash_and_resume, verify_no_skip_no_repeat,
                                    verify_resume_matched_expected)
from pipeline.evidence import build_evidence, write_evidence
from pipeline.ledger import JsonlLedger
from pipeline.packing import packing_utilization
from pipeline.replay import replay_ledger_mode
from pipeline.throughput import compute_performance
from pipeline.train_loop import Config, build_world, materialize_plan


def _build_full_artifacts(tmp_path):
    artifacts_dir = str(tmp_path / "submission_artifacts")
    os.makedirs(os.path.join(artifacts_dir, "manifests"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "ledgers"), exist_ok=True)
    checkpoint_dir = os.path.join(artifacts_dir, "checkpoints")

    # The shipped configuration, so the evidence rows under test are the ones
    # the submitted artifacts actually contain - including the protected-floor
    # override, which a healthy mixture only triggers occasionally.
    config = Config()
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

    for decision in stats["decisions"]:
        opus_ledger.append(decision)

    crash_result = simulate_crash_and_resume(plan, config, consumption, learning, checkpoint_dir, "run-a")

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

    with open(os.path.join(artifacts_dir, "packed_batches.json"), "w") as f:
        json.dump({"pad_id": world["tokenizer"].pad_id, "samples": [
            {"ledger_offset": e["ledger_offset"], "policy": e["sample"]["policy"],
             "seq_len": e["sample"]["seq_len"], "shard_ids": e["sample"]["shard_ids"],
             "token_span_ids": e["sample"]["token_span_ids"], "tokens": e["sample"]["tokens"],
             "position_ids": e["sample"]["position_ids"], "segment_ids": e["sample"]["segment_ids"],
             "loss_mask": e["sample"]["loss_mask"], "loss_mask_hash": e["sample"]["loss_mask_hash"]}
            for e in plan]}, f)

    pack_pct = packing_utilization([e["bin"] for e in plan], seq_len=config.seq_len)

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


def _tamper(artifacts_dir, mutate):
    path = os.path.join(artifacts_dir, "packed_batches.json")
    with open(path) as f:
        packed = json.load(f)
    mutate(packed)
    with open(path, "w") as f:
        json.dump(packed, f)
    return next(r for r in build_evidence(artifacts_dir) if r["requirement"] == "Packing correctness")


def test_packing_evidence_catches_corrupted_loss_mask(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)

    def flip_a_mask_bit(packed):
        packed["samples"][0]["loss_mask"][0] ^= 1

    assert _tamper(artifacts_dir, flip_a_mask_bit)["result"] == "FAIL"


def test_packing_evidence_catches_loss_bearing_padding(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)

    def mark_padding_loss_bearing(packed):
        sample = packed["samples"][0]
        pad_index = len(sample["tokens"]) - 1
        sample["loss_mask"][pad_index] = 1
        sample["tokens"][pad_index] = packed["pad_id"]
        sample["loss_mask_hash"] = hashlib.sha256(bytes(sample["loss_mask"])).hexdigest()

    assert _tamper(artifacts_dir, mark_padding_loss_bearing)["result"] == "FAIL"


def test_packing_evidence_catches_broken_position_reset(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)

    def break_position_reset(packed):
        sample = next(s for s in packed["samples"] if len(s["shard_ids"]) > 1)
        boundary = sample["segment_ids"].index(1)
        sample["position_ids"][boundary] = 99

    assert _tamper(artifacts_dir, break_position_reset)["result"] == "FAIL"


def test_packing_evidence_catches_unreconstructable_utilization(tmp_path):
    """If the reported packing number can't be rebuilt from the spans, the
    rubric gives no credit - so the evidence must not give a PASS either."""
    artifacts_dir = _build_full_artifacts(tmp_path)
    perf_path = os.path.join(artifacts_dir, "performance.json")
    with open(perf_path) as f:
        perf = json.load(f)
    perf["packing_utilization_pct"] = perf["packing_utilization_pct"] + 10
    with open(perf_path, "w") as f:
        json.dump(perf, f)
    row = next(r for r in build_evidence(artifacts_dir) if r["requirement"] == "Packing correctness")
    assert row["result"] == "FAIL"


def test_opus_evidence_requires_rejections_and_deferrals(tmp_path):
    """An audit trail of accepted candidates only is not an audit trail."""
    artifacts_dir = _build_full_artifacts(tmp_path)
    opus_path = os.path.join(artifacts_dir, "ledgers", "opus_decisions.jsonl")
    with open(opus_path) as f:
        kept = [l for l in f if json.loads(l)["decision"] in ("accepted", "protected")]
    with open(opus_path, "w") as f:
        f.writelines(kept)
    row = next(r for r in build_evidence(artifacts_dir) if r["requirement"] == "OPUS audit trail")
    assert row["result"] == "FAIL"


def test_firewall_evidence_catches_admitted_blocked_shard(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)
    manifests_path = os.path.join(artifacts_dir, "manifests", "shard_manifests.json")
    with open(manifests_path) as f:
        manifests = json.load(f)
    next(m for m in manifests if m["eval_overlap_status"] != "clear")["admission"] = "admitted"
    with open(manifests_path, "w") as f:
        json.dump(manifests, f)
    row = next(r for r in build_evidence(artifacts_dir) if r["requirement"] == "Evaluation firewall")
    assert row["result"] == "FAIL"


def test_evidence_reflects_induced_failure(tmp_path):
    artifacts_dir = _build_full_artifacts(tmp_path)
    # Deliberately break the crash-recovery invariant to prove evidence isn't hardcoded.
    report_path = os.path.join(artifacts_dir, "ledgers", "crash_resume_report.json")
    with open(report_path, "w") as f:
        json.dump({"no_skip_no_repeat": False, "resume_matched_expected": False, "resume_offset": 0}, f)
    rows = build_evidence(artifacts_dir)
    crash_row = next(r for r in rows if r["requirement"] == "Crash recovery")
    assert crash_row["result"] == "FAIL"
