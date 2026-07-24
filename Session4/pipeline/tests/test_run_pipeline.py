import json
import pyarrow as pa
import pyarrow.parquet as pq
from run_pipeline import run


def _make_corpus(path):
    # A long, non-repetitive paragraph so its shingle set is large and a tiny
    # appended tail leaves the Jaccard well above the 0.8 dedup threshold.
    good = ("ಕರ್ನಾಟಕ ಸರ್ಕಾರವು ಈ ವರ್ಷ ಹೊಸ ಶಿಕ್ಷಣ ನೀತಿಯನ್ನು ಜಾರಿಗೆ ತರಲು ನಿರ್ಧರಿಸಿದೆ ಎಂದು "
            "ಮುಖ್ಯಮಂತ್ರಿ ಅವರು ಇಂದು ವಿಧಾನಸೌಧದಲ್ಲಿ ನಡೆದ ಸುದ್ದಿಗೋಷ್ಠಿಯಲ್ಲಿ ತಿಳಿಸಿದರು ಈ ನೀತಿಯು "
            "ಗ್ರಾಮೀಣ ಪ್ರದೇಶದ ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ಹೆಚ್ಚಿನ ಅನುಕೂಲ ಕಲ್ಪಿಸಲಿದೆ ಮತ್ತು ಡಿಜಿಟಲ್ ಕಲಿಕೆಗೆ "
            "ಆದ್ಯತೆ ನೀಡಲಿದೆ ಶಾಲೆಗಳಲ್ಲಿ ಆಧುನಿಕ ಪ್ರಯೋಗಾಲಯ ಮತ್ತು ಗ್ರಂಥಾಲಯ ಸೌಲಭ್ಯ ಒದಗಿಸಲಾಗುವುದು "
            "ಎಂದು ಅಧಿಕಾರಿಗಳು ಮಾಹಿತಿ ನೀಡಿದ್ದಾರೆ ಈ ಕುರಿತ ಆದೇಶ ಶೀಘ್ರದಲ್ಲೇ ಹೊರಬೀಳಲಿದೆ")
    d5 = ("ಸಂಪರ್ಕಕ್ಕಾಗಿ ram@example.com ಗೆ ಬರೆಯಿರಿ. ಮುಂಬೈನಲ್ಲಿ ಇಂದು ಭಾರೀ ಮಳೆ "
          "ಸುರಿದ ಪರಿಣಾಮ ಸಂಚಾರ ವ್ಯವಸ್ಥೆ ಸಂಪೂರ್ಣ ಅಸ್ತವ್ಯಸ್ತಗೊಂಡಿತು ಜನಸಾಮಾನ್ಯರು "
          "ತೀವ್ರ ತೊಂದರೆ ಅನುಭವಿಸಿದರು ಎಂದು ವರದಿಗಳು ತಿಳಿಸಿವೆ ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ಶೀಘ್ರ ಲಭ್ಯ")
    rows = [
        ("d1", good),
        ("d2", good + " ಹೆಚ್ಚಿನ ವಿವರ ಪ್ರಕಟ"),                    # near-dup of d1 -> dropped
        ("d3", "this document is english only and should be dropped by langid " * 4),
        ("d4", "ಸಣ್ಣ"),                                          # too short -> dropped
        ("d5", d5),                                                # PII masked, kept
    ]
    pq.write_table(pa.table({"doc_id": [r[0] for r in rows],
                             "text": [r[1] for r in rows]}), path)


class StubNER:
    def __call__(self, text):
        return []          # no names in the synthetic set


def test_run_produces_valid_stats_and_reduces(tmp_path):
    inp = tmp_path / "mini.parquet"
    _make_corpus(str(inp))
    out = tmp_path / "out"
    stats = run(str(inp), str(out), use_ner=True, ner_pipe=StubNER())

    # schema
    for key in ("baseline", "stages", "pii", "final", "manifest", "meta"):
        assert key in stats
    assert len(stats["stages"]) == 6            # normalize..decontam (manifest is separate)

    # monotonic reduction: final docs < baseline docs
    assert stats["final"]["docs"] < stats["baseline"]["docs"]

    # d2 (near-dup), d3 (english), d4 (tiny) gone; d1 and d5 survive
    cleaned = pq.read_table(str(out / "cleaned_corpus.parquet")).to_pydict()
    assert "d1" in cleaned["doc_id"] and "d5" in cleaned["doc_id"]
    assert "d3" not in cleaned["doc_id"] and "d4" not in cleaned["doc_id"]
    assert "d2" not in cleaned["doc_id"]

    # PII masked in d5
    d5 = cleaned["text"][cleaned["doc_id"].index("d5")]
    assert "[EMAIL]" in d5 and "ram@example.com" not in d5

    # files written and manifest valid
    assert (out / "manifest.json").exists()
    saved = json.loads((out / "stats.json").read_text())
    assert saved["final"]["tokens"] <= saved["baseline"]["tokens"]
    assert stats["manifest"]["license"] == "ODC-BY-1.0"
