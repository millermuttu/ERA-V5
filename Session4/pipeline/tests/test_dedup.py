from dedup import shingles, minhash_signature, find_duplicates
import numpy as np


def test_shingles_and_signature_shape():
    sh = shingles("ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು", k=3)
    assert len(sh) == 4
    sig = minhash_signature(sh)
    assert sig.shape == (128,) and sig.dtype == np.int64


def test_identical_docs_have_identical_signatures():
    a = minhash_signature(shingles("ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು"))
    b = minhash_signature(shingles("ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು"))
    assert np.array_equal(a, b)


def test_finds_near_duplicate_keeps_distinct():
    base = ("ಕರ್ನಾಟಕ ಸರ್ಕಾರವು ಹೊಸ ಶಿಕ್ಷಣ ನೀತಿಯನ್ನು ಘೋಷಿಸಿದೆ ಇದು "
            "ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ಅನುಕೂಲಕರವಾಗಿದೆ ಎಂದು ಅಧಿಕಾರಿಗಳು ತಿಳಿಸಿದ್ದಾರೆ")
    near = base + " ಹೆಚ್ಚಿನ ವಿವರ ಶೀಘ್ರದಲ್ಲೇ"       # ~95% overlap
    distinct = ("ಮುಂಬೈನಲ್ಲಿ ಇಂದು ಭಾರೀ ಮಳೆ ಸುರಿಯಿತು ಸಂಚಾರ "
                "ಸಂಪೂರ್ಣವಾಗಿ ಅಸ್ತವ್ಯಸ್ತಗೊಂಡಿತು ಜನರು ಪರದಾಡಿದರು")
    dups = find_duplicates([("a", base), ("b", near), ("c", distinct)], threshold=0.7)
    assert "b" in dups
    assert "a" not in dups and "c" not in dups
