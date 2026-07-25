#!/usr/bin/env python3
"""Stream a ~20M-token slice of CC-News (vblagoje/cc_news), raw & undeduplicated.

CommonCrawl news: syndicated articles repeat across outlets, so this slice has
real exact + near duplicates (verified ~11%), plus emails/phones for PII. Keeps
every document (including duplicates) so the pipeline's dedup stage has work.
"""
import json
import re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from datasets import load_dataset

OUT = Path("Session4/data/raw/ccnews_slice.parquet")
STATS = Path("Session4/data/raw/ccnews_baseline.json")
TARGET_TOKENS = 20_000_000

enc = tiktoken.get_encoding("o200k_base")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)")
PIPE = re.compile(r"\s\|\s")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("vblagoje/cc_news", split="train", streaming=True)
    ids, texts = [], []
    tok = chars = words = 0
    dirt = {"email": 0, "phone": 0, "boiler_pipe": 0, "tiny": 0}
    seen = set()
    exact_dups = 0
    i = 0
    for r in ds:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        ids.append(f"ccnews-{i}")
        texts.append(text)
        tok += len(enc.encode(text))
        chars += len(text)
        words += len(text.split())
        if EMAIL.search(text): dirt["email"] += 1
        if PHONE.search(text): dirt["phone"] += 1
        if PIPE.search(text): dirt["boiler_pipe"] += 1
        if len(text) < 200: dirt["tiny"] += 1
        h = hash(text)
        if h in seen: exact_dups += 1
        else: seen.add(h)
        i += 1
        if tok >= TARGET_TOKENS:
            break
        if i % 2000 == 0:
            print(f"  {i:,} docs, {tok/1e6:.1f}M tokens...", flush=True)

    n = len(texts)
    pq.write_table(pa.table({"doc_id": ids, "text": texts}), OUT)
    stats = {
        "source": "vblagoje/cc_news (CommonCrawl News, English)",
        "docs": n, "tokens_o200k": tok, "words": words, "chars": chars,
        "fertility": round(tok / max(1, words), 3),
        "exact_dup_texts": exact_dups,
        "dirt_doc_pct": {k: round(100 * v / n, 2) for k, v in dirt.items()},
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nsaved {n:,} docs -> {OUT}")


if __name__ == "__main__":
    main()
