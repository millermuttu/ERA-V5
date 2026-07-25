import re

_KANNADA_LO, _KANNADA_HI = 0x0C80, 0x0CFF


def kannada_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    kn = sum(1 for c in letters if _KANNADA_LO <= ord(c) <= _KANNADA_HI)
    return kn / len(letters)


def is_kannada(text, threshold=0.5):
    return kannada_ratio(text) >= threshold


# --- English detection (for Latin-script corpora like CC-News) ---
_EN_STOP = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "on",
            "with", "as", "was", "it", "by", "be", "at", "this", "from", "or",
            "an", "are", "his", "not", "but", "have", "he", "which", "you",
            "has", "were", "they", "their", "said", "will", "would", "who"}
_WORD = re.compile(r"[a-z']+")


def latin_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isascii()) / len(letters)


def english_stopword_ratio(text):
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in _EN_STOP) / len(words)


def is_english(text, min_latin=0.7, min_stop=0.04):
    """English if the script is mostly Latin AND enough English stop-words
    appear -- drops other-script, other-Latin-language, and non-prose junk."""
    return latin_ratio(text) >= min_latin and english_stopword_ratio(text) >= min_stop
