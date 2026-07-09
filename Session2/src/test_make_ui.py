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


def test_missing_tokenizer_file_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(make_ui, "TOKENIZER_PATH", tmp_path / "nope.json")
    with pytest.raises(SystemExit):
        make_ui.load_tokenizer()


def test_wrong_vocab_size_exits(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"base_chars": ["a", "b"], "merges": []}))
    monkeypatch.setattr(make_ui, "TOKENIZER_PATH", bad)
    with pytest.raises(SystemExit):
        make_ui.load_tokenizer()


def test_malformed_merge_tuple_exits_cleanly(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"base_chars": ["a", "b"], "merges": [[1, 2, 3]]}))
    monkeypatch.setattr(make_ui, "TOKENIZER_PATH", bad)
    with pytest.raises(SystemExit):
        make_ui.load_tokenizer()


def test_missing_results_keys_exits(tmp_path, monkeypatch):
    bad = tmp_path / "bad_results.json"
    bad.write_text(json.dumps({"per_language": {}}))  # no "score"
    monkeypatch.setattr(make_ui, "RESULTS_PATH", bad)
    with pytest.raises(SystemExit):
        make_ui.load_results()


def test_committed_html_matches_fresh_build():
    assert make_ui.build_html() == make_ui.OUTPUT_PATH.read_text(encoding="utf-8")
