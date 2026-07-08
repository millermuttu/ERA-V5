import pytest

from bpe_tokenizer import UNK_ID, pretokenize


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
