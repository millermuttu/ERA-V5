from pipeline.throughput import advice, compute_performance, recompute_from_inputs
from pipeline.train_loop import Config


def test_recompute_from_inputs_reproduces_outputs():
    config = Config()
    result = compute_performance(config, pack_pct=82.0, reject_pct=20.0)
    recomputed = recompute_from_inputs(result["inputs"])
    for key in ["tokens_per_step", "useful_after_pack", "useful_after_opus", "worker_factor",
                "shard_penalty", "effective_bandwidth", "useful_tokens_per_sec"]:
        assert result[key] == recomputed[key]


def test_advice_thresholds():
    assert advice(50, 10, 5, 1024) == "packing is the bottleneck"
    assert advice(90, 50, 5, 1024) == "OPUS discarding too much"
    assert advice(90, 10, 30, 1024) == "GPU waiting, fix cache/prefetch/bandwidth"
    assert advice(90, 10, 5, 100) == "bundle into larger immutable objects"
    assert advice(90, 10, 5, 1024) == "throughput looks healthy"
