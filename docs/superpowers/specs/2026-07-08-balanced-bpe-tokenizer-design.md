# Session 2 — Balanced Multilingual BPE Tokenizer (Design)

**Date:** 2026-07-08
**Status:** Approved (Approach A)

## Goal

Train a single BPE tokenizer with a **total vocabulary of 10,000 tokens** on the
Wikipedia "India" article in four languages — English, Hindi, Telugu, Kannada —
such that the per-language fertility ratios are all ≤ 1.2 and as close to each
other as possible.

- **Fertility ratio:** `X_i = (BPE tokens produced on language i's full text) / (whitespace-separated word count of that text)`
- **Constraint:** `X_i ≤ 1.2` for every language
- **Score:** `1000 / (X_max − X_min)` — maximized by keeping the four ratios close.

## Approach

**Codepoint-level BPE with greedy "help the worst language" merge selection.**

1. **Codepoint-level, not byte-level.** Indic characters are 3 bytes in UTF-8;
   byte-level BPE would burn hundreds of merges reassembling characters and skew
   ratios toward English. The base vocabulary is the set of unique Unicode
   codepoints across all four corpora (a few hundred), leaving ~9,500+ slots for
   learned merges.
2. **Word-boundary pre-splitting.** Text is split into words (whitespace
   delimited; whitespace/punctuation retained as their own tokens as needed).
   Merges never cross word boundaries. Word frequency tables make training fast.
3. **Balanced merge loop.** The four corpora stay separate during training. Each
   iteration:
   - compute current fertility `X_i` for each language (cheap: track token
     counts incrementally),
   - pick the language with the **highest** `X_i`,
   - find that language's most frequent adjacent pair and add it as the next
     merge (applied globally to all corpora, since the vocab is shared),
   - repeat until vocab size reaches 10,000.
   This greedily minimizes `X_max − X_min` at every step — a direct optimizer of
   the score formula.
4. **Encoding** applies merges in learned order (standard BPE encode).
   `decode(encode(text)) == text` must hold exactly for all four corpora.

## Data

`download_data.py` fetches plain-text extracts of the "India" article via the
MediaWiki API (`action=query&prop=extracts&explaintext`) from:

- `en.wikipedia.org` → `Session2/data/en_india.txt`
- `hi.wikipedia.org` → `Session2/data/hi_india.txt` (भारत)
- `te.wikipedia.org` → `Session2/data/te_india.txt` (భారతదేశం)
- `kn.wikipedia.org` → `Session2/data/kn_india.txt` (ಭಾರತ)

Titles are resolved per-wiki (the script follows language links from the English
article rather than hard-coding translated titles). Downloaded files are
committed so results are reproducible offline.

## Components

| File | Purpose |
|------|---------|
| `Session2/download_data.py` | Fetch and save the four corpora. |
| `Session2/bpe_tokenizer.py` | `BalancedBPETokenizer`: `train(corpora, vocab_size=10000)`, `encode(text)`, `decode(ids)`, `save(path)` / `load(path)` (JSON), `fertility(text)`. |
| `Session2/Session2_BPE_Tokenizer.ipynb` | Narrative: download → corpus stats → train → X₁…X₄ table → score → encode/decode demos. |
| `Session2/tokenizer_10k.json` | Trained tokenizer artifact (vocab + ordered merges). |

## Error handling

- Download: fail loudly with a clear message if an API call fails or an extract
  comes back empty; no silent fallbacks.
- Encode: any character never seen in training maps to a reserved `<unk>` token
  (id 0); decode renders it as `�`. (With train = eval text this path is
  never hit, but it keeps the tokenizer total.)
- Train: assert vocab budget arithmetic (base chars + merges + specials =
  10,000) before starting the merge loop.

## Testing / verification

- Round-trip: `decode(encode(text)) == text` asserted for each full corpus.
- Constraint check: all four `X_i ≤ 1.2` asserted after training.
- Report: table of per-language word count, token count, `X_i`; final score
  `1000 / (X_max − X_min)`.
- Save/load: reloaded tokenizer produces identical ids on a sample text.

## Out of scope

- Additional training pages beyond the four "India" articles.
- Byte-level fallback, regex pre-tokenization à la GPT-2, special tokens beyond
  `<unk>`.
- Library tokenizers (HuggingFace/SentencePiece) — implementation is from
  scratch in pure Python.
