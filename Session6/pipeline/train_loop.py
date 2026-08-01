"""Materializes the full mixture -> OPUS -> packing plan as an ordered list
of committed batch records indexed by ledger_offset, then a deterministic
toy loss model stands in for "training" (no real model at this toy scale).

Loss phase multipliers and lane adjustments are mined from the course's
token-perplexity-heatmap widget (`Session6/reference/mined-numbers.md`,
widget_11).
"""
import random

from pipeline.batch import build_masked_sample
from pipeline.corpus import doc_text, generate_corpus
from pipeline.manifest import build_manifests
from pipeline.mixture import ProtectedFloorTracker, pick_lane, stage_for_fraction
from pipeline.opus import decide
from pipeline.packing import pack_lane
from pipeline.shards import build_shards
from pipeline.tokenizer import build_tokenizer

SEED = 6
MAX_OPUS_ATTEMPTS = 25

PHASE_BANDS = [("early", 0.0, 0.4), ("mid", 0.4, 0.7), ("late", 0.7, 0.85), ("anneal", 0.85, 1.0)]
PHASE_MULT = {"early": 1.35, "mid": 0.95, "late": 0.72, "anneal": 0.62}
LANE_LOSS_ADJ = {"indic": 0.55, "agentic": 0.35}


class Config:
    def __init__(self, gpus=8, micro_batch=2, grad_accum=16, seq_len=128, ckpt_interval=10,
                 n_per_lane=12, total_steps=150, run_branch_id="run-a"):
        self.gpus = gpus
        self.micro_batch = micro_batch
        self.grad_accum = grad_accum
        self.seq_len = seq_len
        self.ckpt_interval = ckpt_interval
        self.n_per_lane = n_per_lane
        self.total_steps = total_steps
        self.run_branch_id = run_branch_id


def build_world(config):
    """Builds the tokenizer + shards + manifests once; shared by plan
    materialization and by the artifacts written to disk in run_demo.py."""
    docs, eval_docs = generate_corpus(n_per_lane=config.n_per_lane)
    all_texts = [doc_text(d) for d in docs + eval_docs]
    tokenizer = build_tokenizer(all_texts)

    training_shards = build_shards(docs, tokenizer)
    eval_shards = build_shards(eval_docs, tokenizer)
    training_manifests = build_manifests(training_shards, license_tier="safe")
    eval_manifests = build_manifests(eval_shards, license_tier="safe")

    admitted_by_lane = {}
    for shard, manifest in zip(training_shards, training_manifests):
        if manifest["admission"] == "admitted":
            admitted_by_lane.setdefault(shard["lane"], []).append(shard)
    for lane in admitted_by_lane:
        admitted_by_lane[lane].sort(key=lambda s: s["shard_id"])

    return {
        "tokenizer": tokenizer,
        "training_shards": training_shards,
        "eval_shards": eval_shards,
        "training_manifests": training_manifests,
        "eval_manifests": eval_manifests,
        "admitted_by_lane": admitted_by_lane,
    }


def phase_for_fraction(fraction):
    fraction = min(max(fraction, 0.0), 0.999999)
    for name, lo, hi in PHASE_BANDS:
        if lo <= fraction < hi:
            return name
    return PHASE_BANDS[-1][0]


def toy_loss(step, lane, phase):
    base = 3.0 * (0.99 ** step)
    adj = LANE_LOSS_ADJ.get(lane, 0.0)
    if lane == "indic" and phase == "anneal":
        adj = 0.0
    return round(base * PHASE_MULT[phase] + adj, 4)


def materialize_plan(world, config, stats=None):
    """Deterministically builds the ordered list of committed batch records
    a full uninterrupted run would process, indexed by ledger_offset.

    If `stats` (a dict) is given, it is filled in-place with
    `decision_counts` - one entry per step's *final* verdict (the retry
    loop below is an internal determinism mechanism, not a distinct
    candidate each time, so only the final decision counts toward the
    real OPUS keep/reject rate) - so callers can compute a real OPUS
    accept/reject rate without a second pass over the corpus."""
    if stats is not None:
        stats["decision_counts"] = {}
    rng = random.Random(SEED)
    tracker = ProtectedFloorTracker()
    admitted_by_lane = world["admitted_by_lane"]
    tokenizer = world["tokenizer"]
    lane_cursor = {lane: 0 for lane in admitted_by_lane}
    shard_repeat_count = {}

    plan = []
    for step in range(config.total_steps):
        fraction = step / config.total_steps
        stage = stage_for_fraction(fraction)
        phase = phase_for_fraction(fraction)

        lane = pick_lane(tracker, stage, rng)
        lane_under_floor = tracker.lane_under_floor() == lane

        lane_shards = admitted_by_lane[lane]
        cursor = lane_cursor[lane]
        shard = lane_shards[cursor % len(lane_shards)]
        lane_cursor[lane] = cursor + 1

        decision = None
        for attempt in range(MAX_OPUS_ATTEMPTS):
            candidate_id = f"step-{step}-{attempt}-{lane}-{shard['shard_id']}"
            decision = decide(candidate_id, [shard], lane, stage, model_age=step,
                               lane_under_floor=lane_under_floor)
            if decision["decision"] in ("accepted", "protected"):
                break

        if stats is not None:
            stats["decision_counts"][decision["decision"]] = (
                stats["decision_counts"].get(decision["decision"], 0) + 1)

        if decision["decision"] not in ("accepted", "protected"):
            continue

        bins = pack_lane(lane, [shard], seq_len=config.seq_len, eos_id=tokenizer.eos_id)
        sample = build_masked_sample(bins[0], seq_len=config.seq_len, pad_id=tokenizer.pad_id)

        tracker.record(lane, shard["token_count"])
        repeat_count = shard_repeat_count.get(shard["shard_id"], 0)
        shard_repeat_count[shard["shard_id"]] = repeat_count + 1

        plan.append({
            "ledger_offset": len(plan),
            "global_step": step,
            "lane": lane,
            "stage": stage,
            "phase": phase,
            "shard": shard,
            "opus_decision": decision,
            "sample": sample,
            "candidate_batch_id": decision["candidate_batch_id"],
            "avg_token_loss": toy_loss(step, lane, phase),
            "repeated_pass": repeat_count,
        })
    return plan


def run_plan(plan, config, ledger, learning_ledger, checkpoint_dir, run_branch_id, start=0, end=None,
             opus_ledger=None):
    """Consumes plan[start:end]: writes consumption + learning ledger events
    and saves a checkpoint (bound to that batch's own ledger_offset) every
    `ckpt_interval` commits."""
    from pipeline.checkpoint import build_checkpoint, save_checkpoint
    from pipeline.ledger import batch_committed_event, checkpoint_bound_event, learning_ledger_event

    end = len(plan) if end is None else end
    for entry in plan[start:end]:
        offset = entry["ledger_offset"]
        is_ckpt_step = (offset + 1) % config.ckpt_interval == 0
        checkpoint_id = f"ckpt-{run_branch_id}-{offset}" if is_ckpt_step else None

        ledger.append(batch_committed_event(
            offset, run_branch_id, entry["global_step"], entry["sample"], entry["opus_decision"],
            entry["lane"], entry["stage"], checkpoint_id=checkpoint_id,
        ))
        if opus_ledger is not None:
            opus_ledger.append(entry["opus_decision"])
        learning_ledger.append(learning_ledger_event(
            shard_id=entry["shard"]["shard_id"], lane=entry["lane"], stage_phase=entry["phase"],
            avg_token_loss=entry["avg_token_loss"], loss_delta=0.0,
            gradient_norm=round(1.0 + entry["avg_token_loss"] * 0.1, 4),
            opus_score=entry["opus_decision"]["score"], tokens_consumed=entry["shard"]["token_count"],
            global_step=entry["global_step"], repeated_pass=entry["repeated_pass"],
        ))

        if is_ckpt_step:
            checkpoint = build_checkpoint(
                checkpoint_id, entry["global_step"], offset, run_branch_id,
                model_state={"step": entry["global_step"], "loss": entry["avg_token_loss"]},
                optimizer_state={"lr": 3e-4, "step": entry["global_step"]},
                rng_state=f"rng-state-{offset}",
            )
            save_checkpoint(checkpoint, checkpoint_dir)
            ledger.append(checkpoint_bound_event(
                checkpoint_id, offset, run_branch_id, checkpoint["model_state_hash"],
                checkpoint["optimizer_state_hash"], checkpoint["rng_state"],
            ))
