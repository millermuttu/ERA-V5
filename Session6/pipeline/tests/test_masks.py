from pipeline.batch import attends, build_masked_sample
from pipeline.corpus import doc_text, generate_corpus
from pipeline.packing import pack_structure_preserving
from pipeline.shards import build_shards
from pipeline.tokenizer import build_tokenizer


def _agentic_sample(seq_len=64):
    docs, eval_docs = generate_corpus(n_per_lane=4)
    texts = [doc_text(d) for d in docs + eval_docs]
    tok = build_tokenizer(texts)
    agentic_docs = [d for d in docs if d["lane"] == "agentic"]
    shards = build_shards(agentic_docs, tok)
    bins = pack_structure_preserving(shards, seq_len=seq_len)
    sample = build_masked_sample(bins[0], seq_len=seq_len, pad_id=tok.pad_id)
    return sample, tok


def test_loss_mask_excludes_padding():
    sample, tok = _agentic_sample()
    for tok_id, mask in zip(sample["tokens"], sample["loss_mask"]):
        if tok_id == tok.pad_id:
            assert mask == 0


def test_loss_mask_excludes_non_loss_bearing_roles():
    # user/observation turns are not loss-bearing per corpus.py LOSS_BEARING_ROLES.
    # Look only at real tokens - padding is all-zero and would satisfy a naive
    # "0 in loss_mask" check on its own.
    sample, _ = _agentic_sample()
    span_len = sum(int(e) - int(s) for _, s, e in
                   (span.rsplit(":", 2) for span in sample["token_span_ids"]))
    real = sample["loss_mask"][:span_len]
    assert 0 in real, "no non-loss-bearing turn was masked out"
    assert 1 in real, "no loss-bearing turn survived the mask"


def test_position_ids_reset_per_document():
    docs, eval_docs = generate_corpus(n_per_lane=4)
    texts = [doc_text(d) for d in docs + eval_docs]
    tok = build_tokenizer(texts)
    general_docs = [d for d in docs if d["lane"] == "general"][:2]
    shards = build_shards(general_docs, tok)
    bin_ = {"policy": "test", "docs": [
        {"shard_id": shards[0]["shard_id"], "tokens": [1, 2, 3], "roles": ["response"] * 3,
         "start": 0, "end": 3},
        {"shard_id": shards[1]["shard_id"], "tokens": [4, 5], "roles": ["response"] * 2,
         "start": 0, "end": 2},
    ]}
    sample = build_masked_sample(bin_, seq_len=10, pad_id=tok.pad_id)
    assert sample["position_ids"][:5] == [0, 1, 2, 0, 1]
    assert sample["segment_ids"][:5] == [0, 0, 0, 1, 1]
    assert sample["token_span_ids"] == [f"{shards[0]['shard_id']}:0:3",
                                        f"{shards[1]['shard_id']}:0:2"]


def test_attention_blocked_across_document_segments():
    segment_ids = [0, 0, 0, 1, 1]
    assert attends(segment_ids, 4, 3) is True
    assert attends(segment_ids, 3, 2) is False
    assert attends(segment_ids, 2, 0) is True
    assert attends(segment_ids, 1, 4) is False  # non-causal (future token)
