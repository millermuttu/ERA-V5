import re

DEFAULT_THRESHOLDS = {
    "min_chars": 200,
    "max_symbol_ratio": 0.25,
    "max_boiler_line_ratio": 0.5,
    "max_dup_line_ratio": 0.3,
    "min_mean_word_len": 1.2,
    "max_mean_word_len": 40.0,
}

# A line with two or more pipe separators reads like a nav/boilerplate strip.
_NAV_LINE = re.compile(r".+\|.+\|.+")
# Indic combining marks and letters live in these blocks (Devanagari..Malayalam
# covers Kannada 0C80-0CFF). Marks are category Mn/Mc so `isalnum()` is False;
# treating them as letters is what makes the filter Indic-aware.
_INDIC_LO, _INDIC_HI = 0x0900, 0x0D7F


def _is_letterlike(ch, indic_aware):
    if ch.isalnum():
        return True
    if indic_aware and _INDIC_LO <= ord(ch) <= _INDIC_HI:
        return True
    return False


def quality_check(text, thresholds=None, indic_aware=True):
    t = thresholds or DEFAULT_THRESHOLDS
    reasons = []
    s = text.strip()

    if len(s) < t["min_chars"]:
        reasons.append("too_short")

    total = len(s) or 1
    symbols = sum(1 for c in s if not _is_letterlike(c, indic_aware) and not c.isspace())
    if symbols / total > t["max_symbol_ratio"]:
        reasons.append("too_many_symbols")

    lines = [ln for ln in s.split("\n") if ln.strip()]
    if lines:
        nav = sum(1 for ln in lines if _NAV_LINE.match(ln))
        if nav / len(lines) > t["max_boiler_line_ratio"]:
            reasons.append("boilerplate")
        dup = len(lines) - len(set(lines))
        if dup / len(lines) > t["max_dup_line_ratio"]:
            reasons.append("dup_lines")

    words = s.split()
    if words:
        mean_wl = sum(len(w) for w in words) / len(words)
        if not (t["min_mean_word_len"] <= mean_wl <= t["max_mean_word_len"]):
            reasons.append("word_len")

    return (len(reasons) == 0, reasons)
