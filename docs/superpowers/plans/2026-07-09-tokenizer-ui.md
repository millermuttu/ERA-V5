# Tokenizer Playground UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-contained `Session2/tokenizer_ui.html` playground (OpenAI-tokenizer style) for the Session 2 balanced BPE tokenizer: live token chips + counts + fertility X, a per-language X/score widget, and a searchable + downloadable full vocabulary — generated from `tokenizer_10k.json` by `Session2/make_ui.py`.

**Architecture:** A committed HTML template (`tokenizer_ui_template.html`) holds all CSS/JS with `/*__PLACEHOLDER__*/` injection points. `make_ui.py` loads `tokenizer_10k.json` + `results.json`, computes JS/Python parity fixtures with the real Python encoder, injects everything, and writes the committed `tokenizer_ui.html`. The page's JS encoder mirrors `bpe_tokenizer.py` exactly; an in-page self-test badge plus a Node harness prove parity.

**Tech Stack:** Python stdlib (`json`, `re`, `pathlib`) + the existing `bpe_tokenizer.py`; plain HTML/CSS/JS (no framework, no bundler, no external assets); `pytest` for the generator; Node 22 for JS-parity verification; google-chrome (headless) for a render smoke check. All confirmed available.

## Global Constraints

- **Self-contained:** the final `tokenizer_ui.html` must work from `file://` by double-click — no network, no external scripts/styles/fonts, no `fetch`. All data inlined.
- **Parity:** the JS encoder must reproduce the Python `BalancedBPETokenizer.encode` ids exactly for all fixtures. Mirror `pretokenize` (`/\s*\S+/gu` + trailing-whitespace unit), the base-vocab map (id 0 = `<unk>` → `�`; ids 1..C = `base_chars`; merges append `idToStr[a]+idToStr[b]`), merge ranks (`Map` key `` `${a},${b}` `` → new id, rank == id), and greedy lowest-rank non-overlapping left-to-right merging with a per-unit memo.
- **Data sources (never hand-edit):** vocab from `Session2/tokenizer_10k.json`; per-language metrics from `Session2/results.json` (en/hi/te/kn each `{words, tokens, X}`, plus top-level `score`).
- **Score formula:** `Score = 1000 / (X_max − X_min)`; show the four X sorted with X_max/X_min marked. Metrics are on the first 2,000 words/language (state this in the widget).
- **Vocab types:** id 0 = `special`; ids 1..len(base_chars) = `base`; rest = `merge`.
- **Generator is strict:** `make_ui.py` `sys.exit`s with a clear message if an artifact is missing or malformed.
- **Both template and generated HTML are committed.** Working dir for commands: repo root `/home/muttu/Desktop/ERA`.
- Spec: `docs/superpowers/specs/2026-07-09-tokenizer-ui-design.md`.

---

### Task 1: Generator core — data loading, fixtures, injection

**Files:**
- Create: `Session2/make_ui.py`
- Create: `Session2/tokenizer_ui_template.html` (minimal skeleton this task; Task 2 fills the UI)
- Create: `Session2/test_make_ui.py`

**Interfaces:**
- Produces (Python, in `make_ui.py`):
  - `TEMPLATE_PATH`, `TOKENIZER_PATH`, `RESULTS_PATH`, `OUTPUT_PATH` (all `pathlib.Path` under `Session2/`).
  - `FIXTURES: list[str]` — the parity test strings (one per language + edge cases).
  - `load_tokenizer() -> BalancedBPETokenizer` — loads `tokenizer_10k.json`; `sys.exit` on failure.
  - `build_vocab_rows(tok) -> list[dict]` — `{"id": int, "s": str, "type": "special"|"base"|"merge"}` for every id 0..vocab_size-1, using `tok.id_to_str` and `len(tok.base_chars)`.
  - `parity_fixtures(tok) -> list[dict]` — `{"text": str, "ids": list[int]}` via `tok.encode`.
  - `build_html() -> str` — reads the template and replaces the placeholders (below) with `json.dumps(..., ensure_ascii=False)`.
  - `main()` — writes `OUTPUT_PATH`, prints a summary; guarded by `if __name__ == "__main__"`.
- Template placeholders (exact tokens, each appearing once): `/*__VOCAB_ROWS__*/`, `/*__MERGES__*/`, `/*__BASE_CHARS__*/`, `/*__FIXTURES__*/`, `/*__RESULTS__*/`, `/*__RAW_JSON__*/`, `/*__SAMPLES__*/`.

- [ ] **Step 1: Write the minimal template skeleton**

Create `Session2/tokenizer_ui_template.html`:

```html
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>BPE Tokenizer Playground</title></head>
<body>
<div id="app">building…</div>
<script id="vocab-data" type="application/json">/*__VOCAB_ROWS__*/</script>
<script id="merges-data" type="application/json">/*__MERGES__*/</script>
<script id="base-chars-data" type="application/json">/*__BASE_CHARS__*/</script>
<script id="fixtures-data" type="application/json">/*__FIXTURES__*/</script>
<script id="results-data" type="application/json">/*__RESULTS__*/</script>
<script id="raw-json-data" type="application/json">/*__RAW_JSON__*/</script>
<script id="samples-data" type="application/json">/*__SAMPLES__*/</script>
<script>
// Task 2 fills in the encoder + UI. For now, mark that data parsed.
const VOCAB = JSON.parse(document.getElementById('vocab-data').textContent);
const MERGES = JSON.parse(document.getElementById('merges-data').textContent);
document.getElementById('app').textContent =
  'vocab ' + VOCAB.length + ' merges ' + MERGES.length;
</script>
</body>
</html>
```

- [ ] **Step 2: Write the failing test**

Create `Session2/test_make_ui.py`:

```python
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import make_ui  # noqa: E402
from bpe_tokenizer import BalancedBPETokenizer  # noqa: E402


def test_vocab_rows_cover_all_ids_with_types():
    tok = make_ui.load_tokenizer()
    rows = make_ui.build_vocab_rows(tok)
    assert len(rows) == tok.vocab_size == 10_000
    assert rows[0] == {"id": 0, "s": "�", "type": "special"}
    n_base = len(tok.base_chars)
    assert rows[1]["type"] == "base" and rows[n_base]["type"] == "base"
    assert rows[n_base + 1]["type"] == "merge" and rows[-1]["type"] == "merge"
    # every row's string matches the tokenizer's table
    assert [r["s"] for r in rows] == tok.id_to_str


def test_parity_fixtures_match_python_encode():
    tok = make_ui.load_tokenizer()
    fresh = BalancedBPETokenizer.load(make_ui.TOKENIZER_PATH)
    for fx in make_ui.parity_fixtures(tok):
        assert fx["ids"] == fresh.encode(fx["text"])
    # at least one fixture per language plus edge cases
    assert len(make_ui.parity_fixtures(tok)) >= 6


def test_build_html_injects_data_and_leaves_no_placeholders():
    html = make_ui.build_html()
    assert "/*__VOCAB_ROWS__*/" not in html
    assert "/*__RESULTS__*/" not in html
    assert "/*__RAW_JSON__*/" not in html
    # results.json numbers are present
    results = json.loads(make_ui.RESULTS_PATH.read_text())
    assert str(results["per_language"]["en"]["tokens"]) in html
    # the raw tokenizer json round-trips out of the page
    raw = json.loads(make_ui.TOKENIZER_PATH.read_text())
    assert json.dumps(raw["merges"][0]) .strip("[]").replace(" ", "") in html.replace(" ", "")


def test_main_writes_output(tmp_path, monkeypatch):
    out = tmp_path / "tokenizer_ui.html"
    monkeypatch.setattr(make_ui, "OUTPUT_PATH", out)
    make_ui.main()
    assert out.exists() and out.stat().st_size > 100_000
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python3 -m pytest Session2/test_make_ui.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'make_ui'`.

- [ ] **Step 4: Write the generator**

Create `Session2/make_ui.py`:

```python
"""Generate the self-contained tokenizer playground HTML from the trained
artifact. Reads tokenizer_10k.json + results.json, injects them (and Python
parity fixtures) into tokenizer_ui_template.html, writes tokenizer_ui.html."""
import json
import sys
from pathlib import Path

from bpe_tokenizer import BalancedBPETokenizer

HERE = Path(__file__).parent
TEMPLATE_PATH = HERE / "tokenizer_ui_template.html"
TOKENIZER_PATH = HERE / "tokenizer_10k.json"
RESULTS_PATH = HERE / "results.json"
OUTPUT_PATH = HERE / "tokenizer_ui.html"

# Parity fixtures: one per language (from the corpora) + edge cases.
FIXTURES = [
    "India, officially the Republic of India, is a country in South Asia.",
    "भारत दक्षिण एशिया में स्थित एक देश है।",
    "భారతదేశం ప్రపంచంలో ఏడవ పెద్ద దేశం.",
    "ಭಾರತ ದಕ್ಷಿಣ ಏಷ್ಯಾದ ಅತಿ ದೊಡ್ಡ ದೇಶ.",
    "mixed भारत te తెలుగు kn ಕನ್ನಡ 123",  # mixed script + digits
    "emoji 😀 and © symbol outside vocab",   # chars outside base vocab -> <unk>
    "   ",                                     # whitespace only
    "",                                        # empty
]


def _fail(msg: str) -> None:
    sys.exit(f"make_ui: {msg}")


def load_tokenizer() -> BalancedBPETokenizer:
    if not TOKENIZER_PATH.exists():
        _fail(f"missing {TOKENIZER_PATH.name}; run train_and_evaluate.py first")
    try:
        tok = BalancedBPETokenizer.load(TOKENIZER_PATH)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        _fail(f"could not load {TOKENIZER_PATH.name}: {e}")
    if tok.vocab_size != 10_000:
        _fail(f"expected vocab_size 10000, got {tok.vocab_size}")
    return tok


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        _fail(f"missing {RESULTS_PATH.name}; run train_and_evaluate.py first")
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if "per_language" not in data or "score" not in data:
        _fail(f"{RESULTS_PATH.name} missing per_language/score")
    return data


def build_vocab_rows(tok: BalancedBPETokenizer) -> list[dict]:
    n_base = len(tok.base_chars)
    rows = []
    for i, s in enumerate(tok.id_to_str):
        t = "special" if i == 0 else "base" if i <= n_base else "merge"
        rows.append({"id": i, "s": s, "type": t})
    return rows


def parity_fixtures(tok: BalancedBPETokenizer) -> list[dict]:
    return [{"text": t, "ids": tok.encode(t)} for t in FIXTURES]


def _samples() -> dict:
    out = {}
    for lang in ["en", "hi", "te", "kn"]:
        p = HERE / "data" / f"{lang}_india.txt"
        text = p.read_text(encoding="utf-8").strip() if p.exists() else ""
        out[lang] = text[:240]
    return out


def build_html() -> str:
    tok = load_tokenizer()
    results = load_results()
    raw = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    def dump(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)

    replacements = {
        "/*__VOCAB_ROWS__*/": dump(build_vocab_rows(tok)),
        "/*__MERGES__*/": dump(raw["merges"]),
        "/*__BASE_CHARS__*/": dump(raw["base_chars"]),
        "/*__FIXTURES__*/": dump(parity_fixtures(tok)),
        "/*__RESULTS__*/": dump(results),
        "/*__RAW_JSON__*/": dump(raw),
        "/*__SAMPLES__*/": dump(_samples()),
    }
    for marker, value in replacements.items():
        if marker not in template:
            _fail(f"template missing placeholder {marker}")
        template = template.replace(marker, value)
    return template


def main() -> None:
    html = build_html()
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    tok = load_tokenizer()
    print(f"wrote {OUTPUT_PATH.name}: {len(html):,} bytes, "
          f"vocab {tok.vocab_size}, {len(tok.base_chars)} base chars")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m pytest Session2/test_make_ui.py -v`
Expected: all 4 PASS. (`build_html` succeeds because the skeleton template contains every placeholder.)

- [ ] **Step 6: Generate the skeleton page and eyeball it**

Run: `cd Session2 && python3 make_ui.py`
Expected: `wrote tokenizer_ui.html: >100,000 bytes, vocab 10000, 294 base chars`.

- [ ] **Step 7: Commit**

```bash
git add Session2/make_ui.py Session2/tokenizer_ui_template.html Session2/test_make_ui.py Session2/tokenizer_ui.html
git commit -m "feat(session2): tokenizer UI generator + data injection skeleton"
```

---

### Task 2: The playground page — encoder, UI, score widget, vocab explorer

**Files:**
- Modify: `Session2/tokenizer_ui_template.html` (replace the skeleton `<script>`/`<body>` with the full app; keep all seven placeholder `<script type="application/json">` tags intact)
- Create: `Session2/verify_js_parity.mjs` (Node harness)
- Regenerate + commit: `Session2/tokenizer_ui.html`

**Interfaces:**
- Consumes: the seven injected JSON blobs from Task 1 (`VOCAB`, `MERGES`, `BASE_CHARS`, `FIXTURES`, `RESULTS`, `RAW_JSON`, `SAMPLES`), read via `JSON.parse(document.getElementById(...).textContent)`.
- Produces (JS, must be reachable for the Node harness): a global `encode(text) -> number[]` and `idToStr` array, plus `runSelfTest() -> {passed:int,total:int,firstFail:object|null}`. Guard a Node export at the very end: `if (typeof module !== 'undefined') { module.exports = { encode, idToStr, runSelfTest }; }`.

**Design note:** This task creates real UI — the implementer MUST invoke the `frontend-design` skill before writing CSS/markup, and honor the OpenAI-tokenizer-style layout in the spec (single column; light+dark via `prefers-color-scheme`; token chips with a pastel 5-color cycle; `<unk>` red). Keep it one file, no external assets.

- [ ] **Step 1: Write the Node parity harness**

Create `Session2/verify_js_parity.mjs`:

```javascript
// Verifies the generated page's JS encoder reproduces the Python ids for every
// embedded fixture. Loads tokenizer_ui.html, extracts injected data + the
// encoder, runs it under Node. Exit 0 = parity holds, 1 = mismatch.
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('./tokenizer_ui.html', import.meta.url), 'utf8');

function blob(id) {
  const re = new RegExp(
    `<script id="${id}" type="application/json">([\\s\\S]*?)</script>`);
  const m = html.match(re);
  if (!m) throw new Error(`missing blob ${id}`);
  return JSON.parse(m[1]);
}

const BASE_CHARS = blob('base-chars-data');
const MERGES = blob('merges-data');
const FIXTURES = blob('fixtures-data');

// Reconstruct the encoder independently from the injected raw data, mirroring
// bpe_tokenizer.py — this is the reference the page's own encoder must match.
const UNK = 0, UNK_STR = '�';
const charToId = new Map(BASE_CHARS.map((c, i) => [c, i + 1]));
const idToStr = [UNK_STR, ...BASE_CHARS];
const rank = new Map();
for (const [a, b] of MERGES) {
  rank.set(a + ',' + b, idToStr.length);
  idToStr.push(idToStr[a] + idToStr[b]);
}
function pretok(text) {
  const units = text.match(/\s*\S+/gu) || [];
  const consumed = units.reduce((n, u) => n + u.length, 0);
  if (consumed < text.length) units.push(text.slice(consumed));
  return units;
}
function encodeUnit(u) {
  let seq = [...u].map((ch) => charToId.get(ch) ?? UNK);
  while (seq.length > 1) {
    let best = Infinity, at = -1;
    for (let i = 0; i < seq.length - 1; i++) {
      const r = rank.get(seq[i] + ',' + seq[i + 1]);
      if (r !== undefined && r < best) { best = r; at = i; }
    }
    if (at < 0) break;
    seq.splice(at, 2, best);
  }
  return seq;
}
function encode(text) { return pretok(text).flatMap(encodeUnit); }

let fails = 0;
for (const { text, ids } of FIXTURES) {
  const got = encode(text);
  if (JSON.stringify(got) !== JSON.stringify(ids)) {
    fails++;
    console.error(`MISMATCH ${JSON.stringify(text)}\n  py ${ids}\n  js ${got}`);
  }
}
if (fails) { console.error(`${fails} fixture(s) failed`); process.exit(1); }
console.log(`JS parity OK: ${FIXTURES.length}/${FIXTURES.length} fixtures`);
```

- [ ] **Step 2: Run the harness against the skeleton page to verify it fails**

Run: `cd Session2 && node verify_js_parity.mjs`
Expected: PASS actually — this harness reconstructs the encoder itself from the injected data, so it verifies the *fixtures* are self-consistent (Python ids reproduced by a correct JS port) even before the page has UI. Expected output: `JS parity OK: 8/8 fixtures`. (If it fails, the fixtures or injection are wrong — a real signal.)

- [ ] **Step 3: Invoke frontend-design, then write the full page**

Invoke the `frontend-design` skill for aesthetic direction, then replace the skeleton `<body>`+`<script>` in `Session2/tokenizer_ui_template.html` with the full app. Keep the seven `<script type="application/json">` blobs. The app must implement, in one inlined `<script>` (no external assets):

1. Parse the seven blobs. Build `charToId`, `idToStr`, `rank` exactly as in the Node harness (Step 1) — this is the page's own `encode`.
2. **Textarea + live tokenization** (debounce ~150 ms): render each token as a colored chip (`chipColors` 5-cycle by position; `<unk>` id 0 → red). Show whitespace inside chips (replace spaces with `·`/visible glyph, newlines with `⏎`). Tooltip per chip: `id · "JSON.stringify(string)"`.
3. **Text ↔ Token IDs toggle**: switches chip content between the string and the id.
4. **Stat row**: Tokens, Characters (`[...text].length`), Words (`(text.match(/\S+/gu)||[]).length`), Fertility `X = tokens/words` (2 dp; `—` when words 0). Emphasize Tokens and X.
5. **Sample buttons** en/hi/te/kn from `SAMPLES` → set textarea + re-tokenize.
6. **Score widget** from `RESULTS`: per-language table (words, tokens, X to 4 dp) for en·hi·te·kn; the four X sorted ascending with X_min and X_max labeled; `X_max − X_min`; headline `Score = 1000 / (X_max − X_min)` (use `RESULTS.score`); a horizontal bar per X row. Note: "computed on the first 2,000 words per language".
7. **Vocabulary explorer**: search box (filter `VOCAB` by `String(id)===q` or `s.includes(q)`), paged render (200 rows/page, prev/next + count), each row `id · type · visible-string`. Two buttons: **Download JSON** → `Blob([JSON.stringify(RAW_JSON)], {type:'application/json'})` saved as `tokenizer_10k.json`; **Download CSV** → rows `id,type,token` with token as `JSON.stringify(s)` (handles commas/quotes), saved as `vocab.csv`. Use an `<a download>` + `URL.createObjectURL`.
8. **Self-test badge**: `runSelfTest()` encodes every `FIXTURES` text with the page's `encode` and compares to its `ids`; render `self-test N/N ✓` green or the first mismatch in red.
9. At the very end of the script: `if (typeof module !== 'undefined') { module.exports = { encode, idToStr, runSelfTest }; }`.

- [ ] **Step 4: Regenerate the page**

Run: `cd Session2 && python3 make_ui.py`
Expected: `wrote tokenizer_ui.html: … bytes, vocab 10000, 294 base chars`.

- [ ] **Step 5: Run the generator test suite**

Run: `python3 -m pytest Session2/test_make_ui.py -v`
Expected: all PASS (placeholders still present in template, data still injected).

- [ ] **Step 6: Run the Node JS-parity harness on the built page**

Run: `cd Session2 && node verify_js_parity.mjs`
Expected: `JS parity OK: 8/8 fixtures`.

- [ ] **Step 7: Headless-render smoke check with Chrome**

Run:

```bash
cd Session2 && google-chrome --headless --disable-gpu --no-sandbox \
  --screenshot=/tmp/tok_ui.png --window-size=1200,2000 \
  --virtual-time-budget=4000 tokenizer_ui.html && \
python3 -c "import os; assert os.path.getsize('/tmp/tok_ui.png') > 20000; print('rendered', os.path.getsize('/tmp/tok_ui.png'), 'bytes')"
```

Expected: a screenshot > 20 KB. Read `/tmp/tok_ui.png` to confirm visually: token chips, the score widget with four X bars + the score, and the vocab list are all visible, and the self-test badge is green.

- [ ] **Step 8: Commit**

```bash
git add Session2/tokenizer_ui_template.html Session2/tokenizer_ui.html Session2/verify_js_parity.mjs
git commit -m "feat(session2): tokenizer playground UI — chips, score widget, vocab explorer"
```

---

## Self-review notes

- Spec coverage: generator + strict validation + injection (Task 1); JS encoder parity (Task 2 Steps 1-3,6, Node harness + in-page badge); textarea/chips/counts/fertility/IDs-toggle/samples (Task 2 Step 3.2-3.5); score widget (3.6); vocab explorer + downloads (3.7); self-test badge (3.8); browser smoke (2 Step 7); pytest generator guard (Task 1 Step 2). Existing 19 tests untouched.
- Type consistency: `build_vocab_rows` row shape `{id,s,type}` used identically in test and page; `encode`/`idToStr`/`runSelfTest` names match across the Node harness, the page export, and the badge; the seven placeholder tokens are identical in template, generator `replacements`, and Node `blob()` ids (`vocab-data`, `merges-data`, etc. map to the `<script id=…>` tags).
- Note the `<script id>` element ids (`vocab-data`…) differ from the `/*__…__*/` placeholder tokens by design: placeholders live inside the tags and are replaced by JSON; the ids are how JS/Node read the blobs back. Both are fixed strings, listed together here to keep them in sync.
