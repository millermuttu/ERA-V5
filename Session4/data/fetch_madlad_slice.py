#!/usr/bin/env python3
"""Stream a ~20M-token slice of MADLAD-400 noisy Kannada (raw, undeduplicated).

Reads only the head of the 2.5GB gzip stream, stopping at the token budget.
Keeps every document (including exact duplicates) so the pipeline's dedup stage
has real work to do. Assigns index-based doc_ids.
"""
import gzip
import json
import re
import urllib.request
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

URL = ("https://huggingface.co/datasets/allenai/MADLAD-400/resolve/main/"
       "data/kn/kn_noisy_0000.jsonl.gz")
OUT = Path("Session4/data/raw/madlad_kn_slice.parquet")
STATS = Path("Session4/data/raw/madlad_kn_baseline.json")
TARGET_TOKENS = 20_000_000

enc = tiktoken.get_encoding("o200k_base")

ZW = {"​": "ZWSP", "﻿": "BOM", "‌": "ZWNJ", "‍": "ZWJ",
      "‮": "RLO", "�": "REPL"}
HTML_ENT = re.compile(r"&(?:amp|lt|gt|quot|#\d+|nbsp|#x[0-9a-fA-F]+);")
PIPE = re.compile(r"\s\|\s")
LATIN = re.compile(r"[A-Za-z]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "curl/8"})
    resp = urllib.request.urlopen(req, timeout=120)
    gz = gzip.GzipFile(fileobj=resp)

    ids, texts = [], []
    tok = chars = words = 0
    dirt = {k: 0 for k in ["zw", "html_ent", "boiler_pipe", "latin", "email", "phone", "tiny"]}
    zw_chars = 0
    i = 0
    for line in gz:
        rec = json.loads(line)
        text = rec.get("text")
        if not text:
            continue
        ids.append(f"madlad-kn-{i}")
        texts.append(text)
        tok += len(enc.encode(text))
        chars += len(text)
        words += len(text.split())
        zc = sum(text.count(c) for c in ZW)
        zw_chars += zc
        if zc: dirt["zw"] += 1
        if HTML_ENT.search(text): dirt["html_ent"] += 1
        if PIPE.search(text): dirt["boiler_pipe"] += 1
        if LATIN.search(text): dirt["latin"] += 1
        if EMAIL.search(text): dirt["email"] += 1
        if PHONE.search(text): dirt["phone"] += 1
        if len(text.strip()) < 200: dirt["tiny"] += 1
        i += 1
        if tok >= TARGET_TOKENS:
            break
        if i % 2000 == 0:
            print(f"  {i:,} docs, {tok/1e6:.1f}M tokens...", flush=True)

    n = len(texts)
    pq.write_table(pa.table({"doc_id": ids, "text": texts}), OUT)
    stats = {
        "source": "allenai/MADLAD-400 :: data/kn/kn_noisy_0000.jsonl.gz",
        "docs": n, "tokens_o200k": tok, "words": words, "chars": chars,
        "fertility": round(tok / max(1, words), 3),
        "zero_width_chars": zw_chars,
        "dirt_doc_pct": {k: round(100 * v / n, 2) for k, v in dirt.items()},
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nsaved {n:,} docs -> {OUT}")


if __name__ == "__main__":
    main()
