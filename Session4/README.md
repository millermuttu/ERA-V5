# Session 4 — Data Cleaning & Deduplication (Kannada)

Applying Session 4's cleaning strategies **for real** to a raw Kannada web crawl,
then reporting the result in one self-contained widget.

**Deliverable:** [`index.html`](index.html) — a self-contained cleaning report
(also at [`widget/index.html`](widget/index.html)). Deploy the `widget/` folder to Netlify.

## The 8 strategies

Session 4 builds **8** cleaning strategies (extraction is a 9th stage inherited
from Session 3). Seven are applied for real to this corpus; format discipline is a
conversation-data strategy shown as a demo.

1. **Text normalization** — NFC, strip zero-width/control noise, unescape HTML
   entities, collapse whitespace — while **keeping** the Brahmic joiners (ZWNJ/ZWJ).
2. **Format discipline** (ghost-tags) — one canonical special-token format *(demo)*.
3. **Quality filtering** — heuristic cascade with Indic-aware thresholds.
4. **Deduplication** — from-scratch MinHash + LSH near-duplicate detection.
5. **Language-ID validation** — Kannada-script ratio, not folder trust.
6. **PII removal** — regex identifiers (real) + name NER *(demo — see below)*.
7. **Decontamination** — eval-set fingerprints + canary strings.
8. **Reproducibility & manifest** — deterministic content hash + provenance manifest.

## Dataset

- **Source:** `allenai/MADLAD-400`, `data/kn/kn_noisy_0000.jsonl.gz` — the raw,
  minimally-filtered ("noisy") Kannada split. License **ODC-BY-1.0**.
- **Slice:** first documents streamed until **20M o200k tokens** (18,020 docs).
- **Why not Sangraha:** Session 3's `sangraha/unverified/kan` turned out already
  clean + deduplicated (only 1.5% removed), so we switched to a genuinely raw crawl
  that actually exercises the pipeline.

## Results (measured, from `data/cleaned/stats.json`)

| Stage | Docs | Tokens | Removed |
|-------|------|--------|---------|
| baseline | 18,020 | 20.0M | — |
| normalize | 18,020 | 19.9M | 24,736 zero-width chars, 4,875 docs cleaned |
| language-ID | 17,606 | 19.5M | −414 docs (2.3%) |
| quality | 13,023 | 13.9M | **−4,583 docs (28.4%)** — boilerplate |
| dedup | 13,023 | 13.9M | 0 — already deduplicated (verified) |
| PII | 13,023 | 13.9M | masked 125 emails / 454 phones / 649 urls / 10 ips |
| decontam | 13,023 | 13.9M | 0 contamination found |
| **final** | **13,023** | **13.9M** | **30.4% of raw tokens removed** |

**Notes on two honest findings:**
- **Dedup = 0**: every well-maintained public corpus (MADLAD-400, Sangraha, …) is
  already globally deduplicated. Running MinHash+LSH independently and finding 0 is
  the *correct* outcome — dedup is a check you run, not a claim you trust.
- **Names as demo**: non-gated multilingual NER models mangle Kannada script
  (fragments + false positives), so real name masking would degrade the corpus.
  Structured identifiers (regex) are fully real; real Kannada NER needs the gated
  `ai4bharat/IndicNER`.

## Reproduce

```bash
python Session4/data/fetch_madlad_slice.py     # stream 20M-token slice
python Session4/pipeline/run_pipeline.py        # clean -> data/cleaned/{corpus,manifest,stats}
python Session4/widget/build_widget.py          # inject stats.json -> widget/index.html
python -m pytest Session4/pipeline/tests/ -q     # 21 tests
```

Raw + cleaned corpus data lives under `data/` (gitignored). The pipeline stages are
pure, independently tested functions in `pipeline/`.
