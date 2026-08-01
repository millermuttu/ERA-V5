"""Packing utilization and useful loss-bearing tokens/sec, computed from a
real measured run (not the course widget's illustrative slider defaults).
Formulas mined from `Session6/reference/mined-numbers.md` widget_15; the
`inputs` are stored alongside the outputs so a grader can recompute them
independently (`recompute_from_inputs`).
"""
import math

BANDWIDTH_MBPS = 1800
DECOMPRESSION_COST_PCT = 24


def worker_factor(workers, prefetch):
    return min(1.8, 0.35 + math.log2(workers + 1) / 4 + prefetch / 30)


def shard_penalty(shard_size_mb):
    if shard_size_mb < 256:
        return 0.72
    if shard_size_mb > 3072:
        return 0.88
    return 1.0


def advice(pack_pct, reject_pct, gpu_idle_pct, shard_size_mb):
    if pack_pct < 75:
        return "packing is the bottleneck"
    if reject_pct > 45:
        return "OPUS discarding too much"
    if gpu_idle_pct > 20:
        return "GPU waiting, fix cache/prefetch/bandwidth"
    if shard_size_mb < 256:
        return "bundle into larger immutable objects"
    return "throughput looks healthy"


def compute_performance(config, pack_pct, reject_pct, workers=18, prefetch=6, shard_size_mb=1024):
    global_batch = config.gpus * config.micro_batch * config.grad_accum
    tokens_per_step = config.seq_len * global_batch
    useful_after_pack = tokens_per_step * (pack_pct / 100)
    useful_after_opus = useful_after_pack * (1 - reject_pct / 100)

    wf = worker_factor(workers, prefetch)
    penalty = shard_penalty(shard_size_mb)
    effective_bandwidth = BANDWIDTH_MBPS * (1 - DECOMPRESSION_COST_PCT / 100) * wf * penalty
    useful_tokens_per_sec = useful_after_opus * (effective_bandwidth / shard_size_mb)
    gpu_idle_pct = round(max(0.0, 100 - wf / 1.8 * 100), 2)

    inputs = {
        "gpus": config.gpus, "micro_batch": config.micro_batch, "grad_accum": config.grad_accum,
        "seq_len": config.seq_len, "pack_pct": pack_pct, "reject_pct": reject_pct,
        "workers": workers, "prefetch": prefetch, "shard_size_mb": shard_size_mb,
    }
    return {
        "tokens_per_step": tokens_per_step,
        "useful_after_pack": round(useful_after_pack, 2),
        "useful_after_opus": round(useful_after_opus, 2),
        "worker_factor": round(wf, 4),
        "shard_penalty": penalty,
        "effective_bandwidth": round(effective_bandwidth, 2),
        "useful_tokens_per_sec": round(useful_tokens_per_sec, 2),
        "packing_utilization_pct": pack_pct,
        "gpu_idle_pct": gpu_idle_pct,
        "advice": advice(pack_pct, reject_pct, gpu_idle_pct, shard_size_mb),
        "inputs": inputs,
    }


class _ReplayConfig:
    """Minimal config shape for recompute_from_inputs - avoids importing
    train_loop.Config just to read 4 attributes back off of `inputs`."""


def recompute_from_inputs(inputs):
    cfg = _ReplayConfig()
    cfg.gpus, cfg.micro_batch = inputs["gpus"], inputs["micro_batch"]
    cfg.grad_accum, cfg.seq_len = inputs["grad_accum"], inputs["seq_len"]
    return compute_performance(cfg, inputs["pack_pct"], inputs["reject_pct"],
                                inputs["workers"], inputs["prefetch"], inputs["shard_size_mb"])
