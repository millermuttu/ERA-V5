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
