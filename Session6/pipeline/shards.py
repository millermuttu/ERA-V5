"""Content-addressed, immutable tokenized shards.

One shard per source document: shard_id/content_hash are derived from a
hash of the tokenized content, so identical input always yields the same
shard id and any change to the document changes the hash.
"""
import hashlib

from pipeline.cleaning import has_canary, mask_structured, overlap_pct
from pipeline.corpus import doc_text


def _content_hash(token_ids):
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "big", signed=False))
    return h.hexdigest()


def build_shard(doc, tokenizer, eval_fps=frozenset()):
    """doc: a corpus.py doc dict (lane, doc_id, segments[, eval-candidate fields]).

    PII is masked *before* tokenization so identifiers never reach a token id,
    and eval overlap / canary hits are measured off the raw text rather than
    read from a declared field.
    """
    raw = doc_text(doc)
    pii_counts = mask_structured(raw)[1]
    segments = [
        {"role": seg["role"], "token_ids": tokenizer.encode(mask_structured(seg["text"])[0])}
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
        # Declared provenance (source contract, not detectable from text):
        "never_train": doc.get("never_train", False),
        "benchmark_derived": doc.get("benchmark_derived", False),
        "benchmark_id": doc.get("benchmark_id"),
        # Measured by pipeline/cleaning.py:
        "benchmark_overlap_pct": overlap_pct(raw, eval_fps),
        "canary_match": has_canary(raw),
        "pii_counts": pii_counts,
        "pii_found": sum(pii_counts.values()),
    }


def build_shards(docs, tokenizer, eval_fps=frozenset()):
    return [build_shard(doc, tokenizer, eval_fps) for doc in docs]
