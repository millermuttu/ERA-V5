# Session 2 — Tokenizer Playground UI (Design)

**Date:** 2026-07-09
**Status:** Approved

> **Restructure (2026-07-09, approved):** `Session2/` is now web-only for
> Netlify upload — the generated page is `Session2/index.html`, and all
> sources, tests, notebook, data, and artifacts live under `Session2/src/`
> (paths below read accordingly; `make_ui.py` writes `../index.html`).

## Goal

A single, self-contained HTML playground for the Session 2 balanced BPE
tokenizer — inspired by `platform.openai.com/tokenizer`. Paste/type text, see it
split into colored token chips with live token/character/word counts and the
assignment's fertility ratio X = tokens/words; see the per-language X ratios and
the self-score in a stats widget; and browse/download the full 10,000-token
vocabulary. Works by double-click from disk (no server, no network), and stays
in sync with the trained tokenizer.

## Approach

**Builder script + self-contained HTML** (mirrors the existing `make_notebook.py`
pattern). `Session2/make_ui.py` reads `tokenizer_10k.json`, ports nothing by hand
— it embeds the vocab/merges as JSON into an HTML template that contains a small
JavaScript re-implementation of the encoder — and writes `Session2/tokenizer_ui.html`.
Both files are committed so the page works straight from a GitHub checkout.

Rejected: (B) `fetch()`ing the JSON at runtime — fails under `file://` CORS, so
it would silently require a local server; (C) hand-pasting the 128 KB JSON —
stale-artifact bugs on every retrain.

## Components

| File | Responsibility |
|------|----------------|
| `Session2/tokenizer_ui_template.html` | Source template: HTML/CSS/JS with `/*__PLACEHOLDER__*/` injection points. Committed, human-editable. |
| `Session2/make_ui.py` | Load + validate `tokenizer_10k.json` and `results.json`; compute parity fixtures with the **Python** tokenizer; inject vocab + fixtures + score data + samples into the template; write `tokenizer_ui.html`. Refuses to emit if an artifact is missing/malformed. |
| `Session2/tokenizer_ui.html` | Generated, committed. The playground: JS encoder + UI. |
| `Session2/test_make_ui.py` | pytest guarding the generator (fixtures match a fresh Python tokenizer; output contains injected data + markers). |

## JS encoder (embedded in the template)

Mirrors `bpe_tokenizer.py` exactly:

- **pretokenize:** `/\s*\S+/gu` matches → units of optional leading whitespace +
  one word; any trailing pure-whitespace remainder appended as its own unit, so
  `units.join('') === text`.
- **base vocab:** `charToId` from `base_chars` (id = index + 1); id 0 = `<unk>`,
  rendered `�`. `idToStr[0] = "�"`, then base chars, then one entry per merge =
  `idToStr[a] + idToStr[b]`.
- **merge ranks:** `Map` keyed `` `${a},${b}` `` → new id (rank == id, monotonic
  with merge order, so lowest-id-first === earliest-merge-first).
- **encode unit:** greedy — repeatedly find the adjacent pair with the lowest
  rank and apply it (non-overlapping, left-to-right, matching `_apply_merge`),
  until no pair has a rank. Per-unit memo cache. Unknown char → id 0.
- **encode text:** concat encoded units.

## UI layout (single column; light + dark via `prefers-color-scheme`)

1. **Header** — title, and vocab stats (10,000 tokens · 294 base codepoints ·
   9,705 merges · 4 languages), read from the injected data.
2. **Sample buttons** — English / हिन्दी / తెలుగు / ಕನ್ನಡ; each loads the first
   sentence(s) of that language's actual corpus into the textarea.
3. **Textarea** — main input; tokenization runs on `input`, debounced ~150 ms.
4. **Stat row** — Tokens, Characters, Words, **Fertility X = tokens/words**
   (2 decimals; `—` when words = 0). Tokens and X emphasized.
5. **Token view** — each token a colored chip (pastel 5-color cycle by position;
   `<unk>` chips red). Whitespace visible inside chips. A **Text ↔ Token IDs**
   toggle switches chip content between the decoded string and the numeric id.
   Hover tooltip on every chip: `id · "quoted string"`.
6. **Score widget** (new) — a panel presenting the assignment metrics from the
   real training run (`results.json`, injected): a small table of per-language
   words / tokens / **X = tokens/words** for en·hi·te·kn, the four X values
   **sorted** with X_max and X_min marked, `X_max − X_min`, and the headline
   **Score = 1000 / (X_max − X_min)**. Each X row shows a bar so the balance is
   visible at a glance. A one-line note states these are computed on the first
   2,000 words per language (the trained-on corpora). This widget is static
   (reflects the committed run), independent of the live textarea.
7. **Vocabulary explorer** (new) — a searchable, virtualized/paged list of all
   10,000 vocab entries: `id`, token string (whitespace made visible), and
   **type** (`special` id 0 · `base` codepoint · `merge`). A search box filters
   by substring of the token or exact id. Two **download** buttons, both
   generating files client-side via `Blob` (no network): **Download JSON**
   (the exact `tokenizer_10k.json` bytes) and **Download vocab CSV**
   (`id,type,token` rows, token JSON-escaped). Full-list rendering is paged
   (e.g. 200 rows/page) to stay responsive with 10k entries.

## Parity self-test (correctness gate)

`make_ui.py` runs the **Python** tokenizer over ~8 fixtures (one per language +
edge cases: char outside base vocab, whitespace-only, mixed-script, empty) and
embeds `{text, expectedIds}` for each. On page load the JS encodes every fixture
and compares to the embedded truth. A badge renders `self-test: N/N ✓` (green) or
a red failure listing the first mismatch — making any JS/Python drift impossible
to miss.

## Error handling

- Char outside the 294-char base vocab → red `<unk>` chip, id 0 (same as Python).
- `make_ui.py`: `sys.exit` with a clear message if `tokenizer_10k.json` is
  absent or fails to load/validate (vocab size, base_chars, merges present).
- No external assets, no network calls — CSP-clean, works offline.

## Testing

- Build-time: the parity fixtures are generated by the real Python encoder.
- Runtime: the embedded self-test badge (JS vs Python truths) must be green.
- **JS parity via Node** (Node 22 available): a harness loads the generated
  page's encoder and injected data under Node and asserts the JS `encode`
  reproduces the Python ids for every fixture — automated, no browser needed.
- **Browser smoke** (google-chrome available): headless screenshot of the built
  page confirms it renders (chips, score widget, vocab list) and the self-test
  badge is green.
- A `pytest` test (`test_make_ui.py`) asserts `make_ui.py` produces an HTML file
  containing the self-test data, the injected vocab, and the score data, and
  that the fixtures' expected ids match a freshly-loaded Python tokenizer
  (guards the generator itself).
- Existing 19 tokenizer tests untouched.

## Out of scope

- Editing/retraining from the UI; multiple tokenizers; file upload.
- Any framework or bundler — plain HTML/CSS/JS only.
- Server deployment (a Gradio/HF-Spaces variant was offered and declined).
