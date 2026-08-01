from pipeline.tokenizer import build_tokenizer, load_tokenizer, save_tokenizer


def test_same_text_same_ids():
    tok = build_tokenizer(["hello world", "foo bar"])
    assert tok.encode("hello world") == tok.encode("hello world")


def test_vocab_change_changes_hash():
    tok1 = build_tokenizer(["hello world"])
    tok2 = build_tokenizer(["hello world foo"])
    assert tok1.tokenizer_hash != tok2.tokenizer_hash


def test_save_load_roundtrip(tmp_path):
    tok = build_tokenizer(["hello world", "goodbye moon"])
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tok, str(path))
    loaded = load_tokenizer(str(path))
    assert loaded.tokenizer_hash == tok.tokenizer_hash
    assert loaded.encode("hello world") == tok.encode("hello world")
