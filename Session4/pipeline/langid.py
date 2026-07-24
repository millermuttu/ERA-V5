_KANNADA_LO, _KANNADA_HI = 0x0C80, 0x0CFF


def kannada_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    kn = sum(1 for c in letters if _KANNADA_LO <= ord(c) <= _KANNADA_HI)
    return kn / len(letters)


def is_kannada(text, threshold=0.5):
    return kannada_ratio(text) >= threshold
