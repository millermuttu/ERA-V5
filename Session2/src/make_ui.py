"""Generate the self-contained tokenizer playground HTML from the trained
artifact. Reads tokenizer_10k.json + results.json, injects them (and Python
parity fixtures) into tokenizer_ui_template.html, writes ../index.html
(the Session2 web entry point, e.g. for Netlify)."""
import json
import sys
from pathlib import Path

from bpe_tokenizer import BalancedBPETokenizer

HERE = Path(__file__).parent
TEMPLATE_PATH = HERE / "tokenizer_ui_template.html"
TOKENIZER_PATH = HERE / "tokenizer_10k.json"
RESULTS_PATH = HERE / "results.json"
OUTPUT_PATH = HERE.parent / "index.html"  # Session2/index.html — the Netlify entry point

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
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as e:
        _fail(f"could not load {TOKENIZER_PATH.name}: {e}")
    if tok.vocab_size != 10_000:
        _fail(f"expected vocab_size 10000, got {tok.vocab_size}")
    return tok


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        _fail(f"missing {RESULTS_PATH.name}; run train_and_evaluate.py first")
    try:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _fail(f"could not load {RESULTS_PATH.name}: {e}")
    if "per_language" not in data or "score" not in data:
        _fail(f"{RESULTS_PATH.name} missing per_language/score")
    for lang in ["en", "hi", "te", "kn"]:
        row = data["per_language"].get(lang)
        if not isinstance(row, dict) or not {"words", "tokens", "X"} <= row.keys():
            _fail(f"{RESULTS_PATH.name} per_language.{lang} missing words/tokens/X")
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
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

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
