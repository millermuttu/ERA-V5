from pii import mask_structured, mask_names


def test_mask_structured_identifiers():
    text = "ಸಂಪರ್ಕಿಸಿ ram.k@example.com ಅಥವಾ +91 9876543210 ಭೇಟಿ https://foo.example/x IP 10.0.0.1"
    out, counts = mask_structured(text)
    assert "[EMAIL]" in out and "[PHONE]" in out and "[URL]" in out and "[IP]" in out
    assert counts["email"] == 1 and counts["phone"] == 1
    assert counts["url"] == 1 and counts["ip"] == 1
    assert "ram.k@example.com" not in out and "9876543210" not in out


def test_mask_names_with_injected_pipeline():
    name = "ರಾಮಪ್ಪ"
    text = f"ಹೆಸರು {name} ಅವರು ಬಂದರು"
    s = text.index(name)
    e = s + len(name)

    class StubPipe:
        def __call__(self, t):
            return [{"entity_group": "PER", "start": s, "end": e, "word": name}]

    out, n = mask_names(text, pipe=StubPipe())
    assert out == "ಹೆಸರು [NAME] ಅವರು ಬಂದರು" and n == 1
