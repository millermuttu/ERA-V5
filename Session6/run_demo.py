#!/usr/bin/env python3
"""Session 6: Training Data Execution System - single-command demo.

Runs the full path (documents -> shards -> manifests -> mixture -> packing
-> batches -> training -> ledgers -> checkpoint -> crash -> resume ->
replay -> fork -> audit -> throughput -> evidence) and writes
submission_artifacts/. See README.md for the reproduce command and
design.md (openspec/changes/session6-data-execution-system/) for the
architecture behind each step.
"""
import json
import os
import shutil
import sys

from pipeline.audit import audit_behind_checkpoint
from pipeline.checkpoint import first_checkpoint, latest_checkpoint
from pipeline.crash_resume import (simulate_crash_and_resume, verify_no_skip_no_repeat,
                                    verify_resume_matched_expected)
from pipeline.evidence import build_evidence, write_evidence
from pipeline.ledger import JsonlLedger
from pipeline.mixture import feasibility_check
from pipeline.packing import pack_lane, packing_utilization
from pipeline.replay import replay_fork_mode, replay_ledger_mode, replay_random_mode
from pipeline.throughput import compute_performance
from pipeline.train_loop import Config, build_world, materialize_plan, run_plan

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_artifacts")


class Logger:
    def __init__(self, path):
        self.f = open(path, "w")

    def log(self, msg):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def passed(self, name):
        self.log(f"[PASS] {name}")

    def close(self):
        self.f.close()


def _reset_artifacts_dir():
    if os.path.exists(ARTIFACTS_DIR):
        shutil.rmtree(ARTIFACTS_DIR)
    for sub in ("manifests", "ledgers", "checkpoints"):
        os.makedirs(os.path.join(ARTIFACTS_DIR, sub), exist_ok=True)


def run_demo():
    _reset_artifacts_dir()
    log = Logger(os.path.join(ARTIFACTS_DIR, "run.log"))
    all_checks_passed = True

    config = Config()
    run_branch_id = config.run_branch_id
    log.log(f"Session6 Training Data Execution System demo - config: gpus={config.gpus} "
            f"micro_batch={config.micro_batch} grad_accum={config.grad_accum} seq_len={config.seq_len} "
            f"ckpt_interval={config.ckpt_interval}")

    world = build_world(config)
    tokenizer = world["tokenizer"]
    with open(os.path.join(ARTIFACTS_DIR, "manifests", "tokenizer.json"), "w") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)

    all_manifests = world["training_manifests"] + world["eval_manifests"]
    with open(os.path.join(ARTIFACTS_DIR, "manifests", "shard_manifests.json"), "w") as f:
        json.dump(all_manifests, f, indent=2)

    tok_ok = all(m["tokenizer_hash"] == tokenizer.tokenizer_hash for m in all_manifests)
    if tok_ok:
        log.passed("tokenizer_hash_verified")
    else:
        all_checks_passed = False
    log.log(f"shards created: {len(world['training_shards'])} training + {len(world['eval_shards'])} eval")
    log.log(f"manifests validated: {len(all_manifests)} manifests written")

    blocked = [m for m in world["eval_manifests"] if m["eval_overlap_status"] == "blocked_or_unknown"]
    if blocked:
        log.passed("eval_shard_blocked")
    else:
        all_checks_passed = False
    log.log(f"evaluation data blocked: {len(blocked)} of {len(world['eval_manifests'])} eval shards")

    supply_by_lane = {lane: sum(s["token_count"] for s in shards)
                       for lane, shards in world["admitted_by_lane"].items()}
    warnings = feasibility_check(config.total_steps * config.seq_len, supply_by_lane)
    with open(os.path.join(ARTIFACTS_DIR, "manifests", "feasibility.json"), "w") as f:
        json.dump(warnings, f, indent=2)
    log.log(f"mixture compiled: supply={supply_by_lane}, feasibility warnings={len(warnings)}")

    stats = {}
    plan = materialize_plan(world, config, stats=stats)
    log.log(f"batches packed: {len(plan)} committed batches planned "
            f"(of {config.total_steps} steps attempted)")
    log.log(f"OPUS decisions recorded: {stats['decision_counts']}")

    consumption = JsonlLedger(os.path.join(ARTIFACTS_DIR, "ledgers", "consumption.jsonl"))
    learning = JsonlLedger(os.path.join(ARTIFACTS_DIR, "ledgers", "learning.jsonl"))
    opus_ledger = JsonlLedger(os.path.join(ARTIFACTS_DIR, "ledgers", "opus_decisions.jsonl"))
    checkpoint_dir = os.path.join(ARTIFACTS_DIR, "checkpoints")

    crash_result = simulate_crash_and_resume(plan, config, consumption, learning, checkpoint_dir, run_branch_id)
    for entry in plan:
        opus_ledger.append(entry["opus_decision"])
    log.log(f"crash simulated: after ledger_offset={crash_result['crash_offset']}")

    no_skip_no_repeat = verify_no_skip_no_repeat(consumption, len(plan))
    resume_matched = verify_resume_matched_expected(consumption, plan, crash_result["resume_offset"])
    if no_skip_no_repeat and resume_matched:
        log.passed("checkpoint_saved")
        log.passed("resume_next_batch_matched")
    else:
        all_checks_passed = False
    log.log(f"run resumed: from ledger_offset={crash_result['resume_offset']}")

    with open(os.path.join(ARTIFACTS_DIR, "ledgers", "crash_resume_report.json"), "w") as f:
        json.dump({"no_skip_no_repeat": no_skip_no_repeat, "resume_matched_expected": resume_matched,
                    "resume_offset": crash_result["resume_offset"],
                    "checkpoint_id": crash_result["checkpoint_id"]}, f, indent=2)

    replay_end = min(10, len(plan))
    ledger_mode_results = replay_ledger_mode(consumption, world["training_shards"], tokenizer, config,
                                              0, replay_end)
    random_mode_result = replay_random_mode(start=0)
    replay_ok = bool(ledger_mode_results) and all(r["matched"] for r in ledger_mode_results)
    if replay_ok:
        log.passed("replay_hash_matched")
    else:
        all_checks_passed = False
    log.log(f"historical stream replayed: ledger-mode over [0,{replay_end}), "
            f"random-mode negative example shifted_start={random_mode_result['shifted_start']}")

    with open(os.path.join(ARTIFACTS_DIR, "ledgers", "replay_report.json"), "w") as f:
        json.dump({"ledger_mode_results": ledger_mode_results, "random_mode": random_mode_result}, f, indent=2)

    final_checkpoint = latest_checkpoint(checkpoint_dir)
    audit_report = audit_behind_checkpoint(consumption, final_checkpoint)
    with open(os.path.join(ARTIFACTS_DIR, "ledgers", "audit_report.json"), "w") as f:
        json.dump(audit_report, f, indent=2)
    log.log(f"audit completed: {audit_report['batch_count']} batches behind checkpoint "
            f"{final_checkpoint['checkpoint_id']}")

    fork_checkpoint = first_checkpoint(checkpoint_dir)
    fork_result = replay_fork_mode(fork_checkpoint["ledger_offset"], run_branch_id)
    fork_start = fork_result["new_start_offset"]
    if fork_start < len(plan):
        fork_ledger = JsonlLedger(os.path.join(ARTIFACTS_DIR, "ledgers", "fork_consumption.jsonl"))
        fork_learning = JsonlLedger(os.path.join(ARTIFACTS_DIR, "ledgers", "fork_learning.jsonl"))
        run_plan(plan, config, fork_ledger, fork_learning, checkpoint_dir, fork_result["run_branch_id"],
                  start=fork_start, end=len(plan))
    with open(os.path.join(ARTIFACTS_DIR, "ledgers", "fork_report.json"), "w") as f:
        json.dump(fork_result, f, indent=2)
    log.log(f"branch forked: {run_branch_id} -> {fork_result['run_branch_id']} at offset {fork_start}")

    all_bins = []
    for lane, shards in world["admitted_by_lane"].items():
        all_bins.extend(pack_lane(lane, shards, seq_len=config.seq_len, eos_id=tokenizer.eos_id))
    pack_pct = packing_utilization(all_bins, seq_len=config.seq_len)
    decisions = stats["decision_counts"]
    total_decisions = sum(decisions.values()) or 1
    reject_pct = 100.0 * (decisions.get("rejected", 0) + decisions.get("deferred", 0)) / total_decisions
    performance = compute_performance(config, pack_pct=pack_pct, reject_pct=reject_pct)
    with open(os.path.join(ARTIFACTS_DIR, "performance.json"), "w") as f:
        json.dump(performance, f, indent=2)
    log.log(f"performance measured: packing_utilization_pct={pack_pct} reject_pct={round(reject_pct, 2)} "
            f"useful_tokens_per_sec={performance['useful_tokens_per_sec']}")

    rows = build_evidence(ARTIFACTS_DIR)
    write_evidence(rows, ARTIFACTS_DIR)
    evidence_failed = [r for r in rows if r["result"] == "FAIL"]
    log.log(f"evidence bundle written: {len(rows)} rows, {len(evidence_failed)} failing")
    if evidence_failed:
        all_checks_passed = False

    log.log("DEMO COMPLETE" if all_checks_passed else "DEMO COMPLETE WITH FAILURES")
    log.close()
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(run_demo())
