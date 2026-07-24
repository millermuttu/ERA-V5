# Session 4 — Kannada Corpus Cleaning + Widget (Design)

Date: 2026-07-24
Status: approved design, pending spec review

## 1. Goal

Complete the Session 4 assignment (§13 of the lesson): find how many cleaning
*strategies* the session lists, pick a 10–100M-token dataset seen in Session 3,
actually apply the cleanups, and build one widget that reports the strategies,
the dataset, what was cleaned (why/how), other concerns addressed, and final
statistics. Deploy the widget to Netlify.

Two deliverables:

1. **A real Python cleaning pipeline** that runs on a 50M-token Kannada slice and
   emits a cleaned corpus, a provenance manifest, and a `stats.json` of measured
   per-stage before/after numbers plus example diffs.
2. **One self-contained HTML widget** (Session 3 aesthetic) that reads those real
   numbers and presents the assignment answers, deployable to Netlify.

## 2. The "how many strategies" answer

**8 cleaning strategies** are built in Session 4 (extraction is a 9th pipeline
stage inherited from Session 3). This matches the §14 definitional list and the
10 reference widgets (widget_1 = pipeline overview; widget_5 + widget_6 = the two
halves of deduplication; the rest = one widget per strategy).

| # | Strategy | Applied to our data? |
|---|----------|----------------------|
| 1 | Text normalization | ✅ applied |
| 2 | Format discipline (ghost-tags) | ⚠️ demo only (conversation-data strategy; our corpus is web text) |
| 3 | Quality filtering | ✅ applied |
| 4 | Deduplication (MinHash/LSH) | ✅ applied (the headline) |
| 5 | Language-ID & validation | ✅ applied |
| 6 | PII removal | ✅ applied |
| 7 | Decontamination | ✅ applied |
| 8 | Reproducibility & manifest | ✅ applied |

So: 7 applied for real, 1 demoed, extraction inherited → the 8-strategy story
stays whole.

## 3. Dataset

- **Source:** `ai4bharat/sangraha`, config `unverified` (raw crawl), language
  `kan` (Kannada), shard `data-0.parquet`. License CC-BY-4.0.
- **Why:** Sangraha is the corpus Session 3 is built around, and the `unverified`
  crawl tier is exactly the "Indic crawl that had no deduplication at any level"
  that Session 4 §6 calls out. Kannada is a Brahmic script, so it exercises the
  joiner-keeping normalization nuance.
- **Slice:** first documents until 50M o200k tokens (mid-range of 10–100M).
  Reproducible from shard 0 by `slice_and_profile.py`.
- **Schema:** `doc_id` (sha256 of raw text), `text` (Kannada).

### Measured baseline (59,603 docs = 50.0M o200k tokens)

| Metric | Value |
|--------|-------|
| Whitespace words | 16.16M |
| Chars | 134.45M |
| UTF-8 bytes | 360.1M |
| Fertility (o200k tok/word) | 3.10 |
| Exact duplicate `doc_id`s | 0 |
| Code-switched (Latin present) | 32.78% |
| Boilerplate nav (` \| `) | 3.96% |
| Multi-space runs | 5.07% |
| Zero-width chars | 40 (16 docs) |
| HTML entities | 15 docs |
| Emails / phones / URLs | 94 / 314 / 377 |
| Tiny docs (<200 chars) | 14 |

Key narrative hooks: fertility 3.10 illustrates §11's "wrong Indic token ratio";
0 exact dupes means **near-dup detection is the real game**; 32.8% code-switch
motivates language-ID + normalization.

## 4. Pipeline (course order, per widget_1 map)

Cleaning happens **before** the content hash (per §2). Each stage is a pure,
independently testable unit that records before/after doc + token counts and a
few example diffs.

1. **Normalize** *(per-doc)* — NFC normalization; strip ZWSP, BOM, RLO/LRO,
   replacement char, other control chars; **keep ZWJ (U+200D) and ZWNJ (U+200C)**
   as legitimate Brahmic joiners; unescape HTML entities; collapse whitespace runs.
2. **Language-ID validation** *(per-doc)* — dependency-free detector: compute the
   ratio of Kannada-block codepoints (U+0C80–U+0CFF) to total letters; drop/flag
   docs below a Kannada threshold. Surfaces the code-switch/mislabel case.
3. **Quality filter** *(per-doc)* — heuristic cascade: min length, mean word
   length band, symbol:word ratio, boilerplate/nav-line ratio, duplicate-line
   ratio. Indic-aware thresholds; the widget explicitly shows an English-tuned
   threshold wrongly failing good Kannada (filter-bias point).
4. **Deduplication** *(corpus)* — MinHash signatures + LSH banding over survivors;
   drop near-duplicates. Exact dupes are already 0, so this measures the near-dup
   removal the Indic crawl never had.
5. **PII removal** *(per-doc)* — regex mask emails, phones, URLs, IPs; **real name
   detection via `ai4bharat/IndicNER`** (MuRIL-based token classifier, GPU) masking
   PER spans. Batched + truncated inference to fit the 4 GB GPU.
6. **Decontamination** *(corpus)* — fingerprint against a held-out eval set
   (FLORES-200 Kannada) and remove overlapping training docs; canary-string demo.
7. **Manifest + reproducibility** — compute deterministic content hash of the
   cleaned text; emit a manifest JSON per §11 (source, license, cleaning-script
   hash, content hash, token count, language breakdown, per-stage removals). A
   re-run produces identical hashes/ids.

**Ghost-tags (strategy 2)** appears as a small in-widget demo using the lesson's
four-format example, since our corpus is not conversation data.

## 5. Data flow

```
kan_slice.parquet (50M tok)
  → normalize → langid → quality → dedup → pii → decontam
  → cleaned_corpus.parquet + manifest.json + stats.json
        → widget/index.html (stats baked in as JSON) → Netlify
```

## 6. Outputs

- `Session4/data/cleaned/cleaned_corpus.parquet` — surviving cleaned docs.
- `Session4/data/cleaned/manifest.json` — provenance manifest (§11 fields).
- `Session4/data/cleaned/stats.json` — the widget's data contract:
  - `baseline`: docs/tokens/words/chars/bytes/fertility.
  - `stages[]`: `{name, docs_in, docs_out, tokens_in, tokens_out, removed_pct,
    indic_concern, example_diffs[]}`.
  - `pii`: counts by type (emails/phones/urls/ips/names).
  - `final`: docs/tokens/fertility after cleaning + total reduction.
  - `manifest`: embedded copy for the manifest viewer.

## 7. Widget (self-contained HTML, Session 3 aesthetic)

Answers every §13 requirement, driven by `stats.json` baked into the page:

- **The 8 strategies** — count + one-line description + the V4 defect each fixes.
- **Dataset picked** — Sangraha `unverified/kan`, why.
- **What was cleaned / why / how** — per stage: real before→after tokens/docs, an
  example diff, and the Indic concern.
- **Surviving-token bar** — collapses across stages using real numbers.
- **Other concerns** — 3.10 fertility, 32.8% code-switch, near-dup-vs-exact-dup,
  keep-the-joiner nuance, filter bias, ghost-tags demo.
- **Final statistics** — before/after totals + downloadable manifest JSON.

Vanilla HTML/CSS/JS, no external dependencies (self-contained, Netlify-ready).
The widget file doubles as `Session4/index.html`.

## 8. Testing (TDD)

Each stage is a pure function with unit tests on crafted inputs:

- **normalize**: a string with ZWSP+BOM+RLO+entity+multispace+ZWNJ → asserts noise
  removed, **ZWJ/ZWNJ preserved**, entity unescaped, whitespace collapsed.
- **langid**: pure-Kannada doc passes; English/romanized doc flagged.
- **quality**: boilerplate/tiny doc dropped; good Kannada doc kept; English-tuned
  threshold demonstrably over-penalizes a good Kannada doc.
- **dedup**: a near-dup pair collides in LSH; a distinct pair does not.
- **pii**: email/phone/URL masked; a PER name masked; surrounding Kannada intact.
- **decontam**: a doc containing an eval fingerprint is removed; a canary is
  detectable.
- **manifest**: same input → identical content hash/ids across two runs; a missing
  required field marks the shard blocked.

## 9. Tech stack

- Python 3.13; `datasets`, `pyarrow`, `pandas` (installed).
- `tiktoken` o200k_base for token counting (installed).
- MinHash/LSH: `datasketch` if present, else a small from-scratch implementation.
- NER: `transformers` + `torch` (installed, CUDA on GTX 1650 Ti 4 GB),
  `ai4bharat/IndicNER`.
- Language-ID: dependency-free script-ratio (no library).
- Widget: vanilla HTML/CSS/JS.
- Deploy: Netlify (drag-drop of `widget/`, or `netlify deploy`).

## 10. Layout

```
Session4/
  reference/            # crawled lesson + 10 reference widgets (done)
  data/                 # gitignored: raw slice, cleaned corpus, manifest, stats.json
  pipeline/             # Python stages + tests
    normalize.py  langid.py  quality.py  dedup.py  pii.py  decontam.py  manifest.py
    run_pipeline.py     # orchestrates stages, writes stats.json
    tests/
  widget/index.html     # self-contained; also copied to Session4/index.html
```

## 11. Open risks / notes

- **GPU memory (4 GB):** IndicNER inference must batch small and truncate long
  docs; benchmark throughput early. If full-corpus NER is too slow, run it on the
  post-dedup survivor set (what actually ships) and document throughput.
- **FLORES Kannada availability:** if the eval set is gated/unavailable, fall back
  to a small hand-built held-out probe + canary strings so the stage still runs.
- **Slice determinism:** the 50M-token cut is defined by document order in shard 0;
  keep the slicing script fixed so numbers are reproducible.
