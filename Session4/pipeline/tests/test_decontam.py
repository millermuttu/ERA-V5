from decontam import (build_eval_fingerprint_set, is_contaminated,
                      has_canary, CANARY)


def test_flags_overlap_only():
    eval_texts = ["ಇದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು"]
    fps = build_eval_fingerprint_set(eval_texts, n=6)
    leak = "ಪೀಠಿಕೆ ಇದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು ಕೊನೆ"
    clean = "ಸಂಪೂರ್ಣ ಬೇರೆ ವಿಷಯ ಇಲ್ಲಿದೆ ಯಾವುದೇ ಹೋಲಿಕೆ ಇಲ್ಲ ಎಂದು ಹೇಳಬಹುದು"
    assert is_contaminated(leak, fps, n=6)
    assert not is_contaminated(clean, fps, n=6)


def test_canary_detection():
    assert has_canary("ರಾಂಡಮ್ " + CANARY + " ಪಠ್ಯ")
    assert not has_canary("ಶುದ್ಧ ಪಠ್ಯ")
