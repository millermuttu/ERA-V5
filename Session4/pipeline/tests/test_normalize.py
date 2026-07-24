from normalize import normalize_text


def test_keeps_brahmic_joiners_strips_noise_and_entities():
    # \u200b ZWSP, \ufeff BOM, \u202e RLO = noise; \u200c ZWNJ + \u200d ZWJ = keep
    dirty = "\u0c85\u200b\ufeff\u202e &amp; \u0cac\u200c\u0cb8\u200d   \u0c95\x07"
    clean, ops = normalize_text(dirty)
    assert "\u200b" not in clean          # ZWSP removed
    assert "\ufeff" not in clean          # BOM removed
    assert "\u202e" not in clean          # RLO removed
    assert "\ufffd" not in clean          # replacement never introduced
    assert "\x07" not in clean            # control char removed
    assert "\u200c" in clean              # ZWNJ preserved
    assert "\u200d" in clean              # ZWJ preserved
    assert "&amp;" not in clean and "&" in clean   # entity unescaped
    assert "   " not in clean             # whitespace collapsed
    assert {"html_unescape", "strip_noise", "strip_control", "collapse_ws"} <= set(ops)


def test_idempotent_on_clean_text():
    clean_once, _ = normalize_text("\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0caa\u0ca0\u0ccd\u0caf")
    clean_twice, ops = normalize_text(clean_once)
    assert clean_once == clean_twice and ops == []
