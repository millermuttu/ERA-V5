from pipeline.corpus import doc_text, generate_corpus
from pipeline.packing import (packing_utilization, pack_best_fit, pack_concat_and_chop,
                               pack_greedy, pack_structure_preserving)
from pipeline.shards import build_shards
from pipeline.tokenizer import build_tokenizer


def _shards(lane, n=6):
    docs, eval_docs = generate_corpus(n_per_lane=n)
    texts = [doc_text(d) for d in docs + eval_docs]
    tok = build_tokenizer(texts)
    lane_docs = [d for d in docs if d["lane"] == lane]
    return build_shards(lane_docs, tok), tok


def test_structure_preserving_never_mixes_documents():
    shards, _ = _shards("agentic")
    bins = pack_structure_preserving(shards, seq_len=128)
    assert all(len(b["docs"]) == 1 for b in bins)


def test_concat_and_chop_token_accounting():
    shards, tok = _shards("general")
    bins = pack_concat_and_chop(shards, seq_len=32, eos_id=tok.eos_id)
    total_in_bins = sum(len(d["tokens"]) for b in bins for d in b["docs"])
    total_input = sum(shard["token_count"] for shard in shards) + len(shards)  # + 1 eos per doc
    assert total_in_bins == total_input


def test_greedy_utilization_in_range():
    shards, _ = _shards("code")
    bins = pack_greedy(shards, seq_len=128)
    util = packing_utilization(bins, seq_len=128)
    assert 0 <= util <= 100


def test_best_fit_utilization_at_least_greedy_bin_count_le():
    shards, _ = _shards("code")
    greedy_bins = pack_greedy(shards, seq_len=128)
    best_fit_bins = pack_best_fit(shards, seq_len=128)
    assert len(best_fit_bins) <= len(greedy_bins)
