from langid import kannada_ratio, is_kannada


def test_pure_kannada_passes_english_fails():
    assert is_kannada("ಕನ್ನಡ ಭಾಷೆ ಒಂದು ಸುಂದರ ಭಾಷೆ")
    assert not is_kannada("this is plain english text")


def test_ratio_between_zero_and_one_for_codeswitch():
    r = kannada_ratio("ಕನ್ನಡ hello world")
    assert 0.0 < r < 1.0


def test_empty_is_zero():
    assert kannada_ratio("12345 !!! ") == 0.0
