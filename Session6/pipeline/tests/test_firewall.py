from pipeline.firewall import check


def _shard(**overrides):
    base = {"never_train": False, "benchmark_overlap_pct": 0.0,
            "canary_match": False, "benchmark_derived": False}
    base.update(overrides)
    return base


def test_clean_shard_passes():
    assert check(_shard())["passed"] is True


def test_never_train_blocks():
    result = check(_shard(never_train=True))
    assert result["passed"] is False
    assert "never_train" in result["reasons"]


def test_overlap_over_threshold_blocks():
    result = check(_shard(benchmark_overlap_pct=61.0))
    assert result["passed"] is False
    assert "benchmark_overlap_exceeded" in result["reasons"]


def test_overlap_under_threshold_passes():
    assert check(_shard(benchmark_overlap_pct=12.0))["passed"] is True


def test_canary_match_blocks():
    assert check(_shard(canary_match=True))["passed"] is False


def test_derived_content_without_never_train_still_blocked():
    result = check(_shard(benchmark_derived=True))
    assert result["passed"] is False
    assert "benchmark_derived" in result["reasons"]
