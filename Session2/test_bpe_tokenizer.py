import pytest

from bpe_tokenizer import UNK_ID, pretokenize, BalancedBPETokenizer


@pytest.mark.parametrize("text", [
    "hello world",
    "  leading and trailing  ",
    "line one\n\nline two\n",
    "नमस्ते   दुनिया",
    "",
    "   ",
])
def test_pretokenize_round_trip(text):
    assert "".join(pretokenize(text)) == text


def test_pretokenize_attaches_leading_whitespace():
    assert pretokenize("a b  c") == ["a", " b", "  c"]


def test_pretokenize_trailing_whitespace_is_own_unit():
    assert pretokenize("ab \n") == ["ab", " \n"]


def test_encode_decode_with_explicit_merges():
    # base ids: ' '=1, 'a'=2, 'b'=3, 'c'=4; merge (2,3) -> id 5 = "ab"
    tok = BalancedBPETokenizer([" ", "a", "b", "c"], [(2, 3)])
    assert tok.vocab_size == 6
    assert tok.encode("ab c") == [5, 1, 4]
    assert tok.decode([5, 1, 4]) == "ab c"


def test_chained_merges():
    # merge (2,3) -> 5 = "ab", merge (5,4) -> 6 = "abc"
    tok = BalancedBPETokenizer([" ", "a", "b", "c"], [(2, 3), (5, 4)])
    assert tok.encode("abc") == [6]
    assert tok.decode([6]) == "abc"


def test_unknown_char_maps_to_unk():
    tok = BalancedBPETokenizer([" ", "a"], [])
    ids = tok.encode("a ω")
    assert ids == [2, 1, UNK_ID]
    assert tok.decode(ids) == "a �"


def test_round_trip_no_merges():
    tok = BalancedBPETokenizer(sorted(set("hello world")), [])
    assert tok.decode(tok.encode("hello world")) == "hello world"


TINY = {
    "en": "the cat sat on the mat the cat sat " * 20,
    "xx": "aba abb aba abb aba " * 20,
}
TINY_BASE = len(set(TINY["en"] + TINY["xx"]))  # unique codepoints across corpora


def test_train_reaches_exact_vocab_size():
    target = 1 + TINY_BASE + 10
    tok = BalancedBPETokenizer.train(TINY, vocab_size=target)
    assert tok.vocab_size == target


def test_train_round_trip_on_training_text():
    tok = BalancedBPETokenizer.train(TINY, vocab_size=1 + TINY_BASE + 10)
    for text in TINY.values():
        assert tok.decode(tok.encode(text)) == text


def test_merges_reduce_token_count():
    tok = BalancedBPETokenizer.train(TINY, vocab_size=1 + TINY_BASE + 10)
    base = BalancedBPETokenizer(tok.base_chars, [])
    text = TINY["en"]
    assert len(tok.encode(text)) < len(base.encode(text))


def test_vocab_size_must_exceed_base_charset():
    with pytest.raises(ValueError):
        BalancedBPETokenizer.train(TINY, vocab_size=5)


def test_merge_loop_helps_worst_language_first():
    corpora = {
        "short": "a b c d " * 50,               # fertility ~2 (space+char units)
        "long": "qqqqqqqq rrrrrrrr " * 50,      # fertility ~9, clearly worst
    }
    n_base = len(set(corpora["short"] + corpora["long"]))
    tok = BalancedBPETokenizer.train(corpora, vocab_size=1 + n_base + 4)
    base = BalancedBPETokenizer(tok.base_chars, [])
    # all 4 merges must have gone to the worst language ("long")
    assert tok.fertility(corpora["long"]) < base.fertility(corpora["long"])
    assert tok.fertility(corpora["short"]) == base.fertility(corpora["short"])


def test_fertility_counts_tokens_per_word():
    tok = BalancedBPETokenizer([" ", "a", "b"], [(2, 3)])  # "ab" -> one token
    assert tok.fertility("ab ab") == 1.5          # 3 tokens / 2 words
    assert tok.fertility("ba ba") == 2.5          # units "ba", " ba" -> 2 + 3 tokens


def test_save_load_identity(tmp_path):
    tok = BalancedBPETokenizer.train(TINY, vocab_size=1 + TINY_BASE + 10)
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BalancedBPETokenizer.load(path)
    sample = "the cat sat aba"
    assert tok2.encode(sample) == tok.encode(sample)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.merges == tok.merges
