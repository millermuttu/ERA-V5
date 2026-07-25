# Session 4 — Data Cleaning & Deduplication (CC-News)

Applying Session 4's cleaning strategies **for real** to a raw, undeduplicated
news crawl, then reporting the result in one self-contained widget.

**Deliverable:** [`index.html`](index.html) — a self-contained cleaning report
(also at [`widget/index.html`](widget/index.html)). Deploy the `widget/` folder to Netlify.

## The 8 strategies

Session 4 builds **8** cleaning strategies (extraction is a 9th stage inherited
from Session 3). Seven are applied for real to this corpus; format discipline is a
conversation-data strategy shown as a demo.

1. **Text normalization** — NFC, strip zero-width/control noise, unescape HTML
   entities, collapse whitespace (joiners preserved for any script).
2. **Format discipline** (ghost-tags) — one canonical special-token format *(demo)*.
3. **Quality filtering** — heuristic cascade (length, symbol ratio, boilerplate,
   duplicate lines, mean word length).
4. **Deduplication** — from-scratch MinHash + LSH near-duplicate detection.
5. **Language-ID validation** — Latin-script ratio + English stop-word density.
6. **PII removal** — regex identifiers (real) + name NER *(demo)*.
7. **Decontamination** — held-out eval fingerprints (13-gram) + canary strings.
8. **Reproducibility & manifest** — deterministic content hash + provenance manifest.

## Dataset

- **Source:** `vblagoje/cc_news` — CommonCrawl News (English). License **CommonCrawl-ToU**.
- **Slice:** first documents streamed until **20M o200k tokens** (38,516 docs).
- **Why CC-News:** curated corpora (Sangraha, MADLAD-400, …) are already globally
  deduplicated, so dedup finds nothing on them. CC-News is **undeduplicated** — the
  same wire story is syndicated across outlets — so near-duplicates, emails and
  phone numbers are genuinely present. Verified up front (probe: 6.8% near-dupes,
  4.5% exact dupes, 6%+ PII) before committing.

## Results (measured, from `data/cleaned/stats.json`)

| Stage | Docs | Tokens | Removed |
|-------|------|--------|---------|
| baseline | 38,516 | 20.0M | — |
| normalize | 38,516 | 20.0M | 378 zero-width chars, 147 docs cleaned |
| language-ID | 38,296 | 19.7M | −220 docs (1.74%) non-English |
| quality | 36,270 | 19.5M | −2,026 docs (0.72%) |
| **dedup** | 31,417 | 16.8M | **−4,853 docs (13.77%)** — syndicated reposts |
| PII | 31,417 | 16.8M | masked 1,424 emails / 1,683 phones / 4,661 urls / 3 ips |
| decontam | 29,489 | 15.9M | −1,928 docs (5.29%) — held-out eval firewall |
| **final** | **29,489** | **15.9M** | **20.6% of raw tokens removed** |

**Notes:**
- **Dedup is the headline**: CommonCrawl news is undeduplicated, so MinHash+LSH
  reclaims ~14% the source never removed — the near-duplicates exact hashing misses.
- **Decontamination** holds out 50 documents as an eval set and removes them plus any
  training doc sharing a verbatim 13-word run (the GPT-3 decontam standard) so
  benchmark scores stay honest.
- **Names as demo**: structured identifiers (regex) are fully real; personal-name NER
  is a demo to keep the run fast and dependency-light.
- **English is char-clean**, so normalization does little here — which stages bite is
  a property of the data. (An earlier Kannada/MADLAD run exercised normalize/langid
  hard but was pre-deduplicated; see git history.)

## Reproduce

```bash
python Session4/data/fetch_ccnews_slice.py     # stream 20M-token slice
python Session4/pipeline/run_pipeline.py        # clean -> data/cleaned/{corpus,manifest,stats}
python Session4/widget/build_widget.py          # inject stats.json -> widget/index.html
python -m pytest Session4/pipeline/tests/ -q     # 22 tests
```

Raw + cleaned corpus data lives under `data/` (gitignored). The pipeline stages are
pure, independently tested functions in `pipeline/`.
