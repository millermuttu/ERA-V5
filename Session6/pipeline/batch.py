"""Batch assembly: loss masks, attention segment ids, and position ids.

Attention is represented as per-token segment ids (Megatron/FlashAttention
"cu_seqlens"-style), not a dense seq_len x seq_len matrix - no real packed-
sequence implementation materializes the dense matrix either, so this is
the minimal *correct* representation, not a toy shortcut.
"""
import hashlib

from pipeline.corpus import LOSS_BEARING_ROLES


def build_masked_sample(bin_, seq_len, pad_id):
    tokens, position_ids, segment_ids, loss_mask, shard_ids, spans = [], [], [], [], [], []
    for seg_idx, doc in enumerate(bin_["docs"]):
        shard_ids.append(doc["shard_id"])
        spans.append(f"{doc['shard_id']}:{doc['start']}:{doc['end']}")
        for pos, (tok, role) in enumerate(zip(doc["tokens"], doc["roles"])):
            tokens.append(tok)
            position_ids.append(pos)
            segment_ids.append(seg_idx)
            loss_mask.append(1 if role in LOSS_BEARING_ROLES else 0)

    pad_segment = len(bin_["docs"])
    while len(tokens) < seq_len:
        tokens.append(pad_id)
        position_ids.append(0)
        segment_ids.append(pad_segment)
        loss_mask.append(0)
    tokens, position_ids = tokens[:seq_len], position_ids[:seq_len]
    segment_ids, loss_mask = segment_ids[:seq_len], loss_mask[:seq_len]

    loss_mask_hash = hashlib.sha256(bytes(loss_mask)).hexdigest()
    return {
        "shard_ids": shard_ids,
        "token_span_ids": spans,
        "seq_len": seq_len,
        "pad_id": pad_id,
        "tokens": tokens,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "loss_mask": loss_mask,
        "loss_mask_hash": loss_mask_hash,
        "policy": bin_["policy"],
    }


def attends(segment_ids, i, j):
    """Causal + same-segment attention rule: token i may attend to token j."""
    return j <= i and segment_ids[i] == segment_ids[j]


def assemble_microbatch(samples):
    return {
        "shard_ids": [s["shard_ids"] for s in samples],
        "tokens": [s["tokens"] for s in samples],
        "position_ids": [s["position_ids"] for s in samples],
        "segment_ids": [s["segment_ids"] for s in samples],
        "loss_mask": [s["loss_mask"] for s in samples],
    }
