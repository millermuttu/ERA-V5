from quality import quality_check, DEFAULT_THRESHOLDS

GOOD = ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ಒಂದು ದೊಡ್ಡ ನಗರವಾಗಿದೆ. "
        "ಇಲ್ಲಿ ಅನೇಕ ತಂತ್ರಜ್ಞಾನ ಕಂಪನಿಗಳು ಕಾರ್ಯ ನಿರ್ವಹಿಸುತ್ತವೆ. "
        "ಈ ನಗರವು ಶಿಕ್ಷಣ ಮತ್ತು ಸಂಶೋಧನೆಗೆ ಹೆಸರುವಾಸಿಯಾಗಿದೆ. ") * 3


def test_good_doc_passes_tiny_doc_fails():
    ok, reasons = quality_check(GOOD)
    assert ok and reasons == []
    ok2, reasons2 = quality_check("ಸಣ್ಣ ಪಠ್ಯ")
    assert not ok2 and "too_short" in reasons2


def test_boilerplate_nav_is_rejected():
    nav = "\n".join(["ಮುಖಪುಟ | ಸುದ್ದಿ | ಕ್ರೀಡೆ | ಸಂಪರ್ಕ"] * 20)
    ok, reasons = quality_check(nav)
    assert not ok and "boilerplate" in reasons


def test_indic_bias_english_tuned_filter_wrongly_fails_good_kannada():
    passed_aware, _ = quality_check(GOOD, indic_aware=True)
    passed_naive, reasons = quality_check(GOOD, indic_aware=False)
    assert passed_aware is True
    assert passed_naive is False and "too_many_symbols" in reasons
