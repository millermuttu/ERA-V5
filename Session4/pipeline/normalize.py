# Session4/pipeline/normalize.py
import html
import re
import unicodedata

# Invisible characters that are noise and must be removed.
# NOTE: U+200C ZWNJ and U+200D ZWJ are intentionally absent -- they are
# legitimate Brahmic joiners and must survive.
_NOISE = ["​", "﻿", "‎", "‏",
          "‪", "‫", "‬", "‭", "‮", "�"]

# Control chars except tab/newline/carriage-return.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")


def normalize_text(text):
    ops = []

    unescaped = html.unescape(text)
    if unescaped != text:
        ops.append("html_unescape")
    text = unescaped

    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        ops.append("nfc")
    text = nfc

    before = text
    for ch in _NOISE:
        text = text.replace(ch, "")
    if text != before:
        ops.append("strip_noise")

    before = text
    text = _CONTROL.sub("", text)
    if text != before:
        ops.append("strip_control")

    before = text
    text = _SPACES.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    if text != before:
        ops.append("collapse_ws")

    return text, ops
