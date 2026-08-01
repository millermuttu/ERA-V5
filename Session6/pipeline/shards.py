"""Content-addressed, immutable tokenized shards.

One shard per source document: shard_id/content_hash are derived from a
hash of the tokenized content, so identical input always yields the same
shard id and any change to the document changes the hash.
"""
import hashlib


def _content_hash(token_ids):
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "big", signed=False))
    return h.hexdigest()


def build_shard(doc, tokenizer):
    """doc: a corpus.py doc dict (lane, doc_id, segments[, eval-candidate fields])."""
    segments = [
        {"role": seg["role"], "token_ids": tokenizer.encode(seg["text"])}
        for seg in doc["segments"]
    ]
    all_ids = [tid for seg in segments for tid in seg["token_ids"]]
    chash = _content_hash(all_ids)
    return {
        "shard_id": f"{doc['lane']}-{chash[:12]}",
        "lane": doc["lane"],
        "doc_ids": [doc["doc_id"]],
        "segments": segments,
        "token_count": len(all_ids),
        "content_hash": chash,
        "tokenizer_hash": tokenizer.tokenizer_hash,
        "never_train": doc.get("never_train", False),
        "benchmark_overlap_pct": doc.get("benchmark_overlap_pct", 0.0),
        "canary_match": doc.get("canary_match", False),
        "benchmark_derived": doc.get("benchmark_derived", False),
        "benchmark_id": doc.get("benchmark_id"),
    }


def build_shards(docs, tokenizer):
    return [build_shard(doc, tokenizer) for doc in docs]
