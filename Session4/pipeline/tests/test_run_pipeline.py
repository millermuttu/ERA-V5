import json
import pyarrow as pa
import pyarrow.parquet as pq
from run_pipeline import run


def _make_corpus(path):
    # A long, non-repetitive paragraph so its shingle set is large and a tiny
    # appended tail leaves the Jaccard well above the 0.8 dedup threshold.
    good = ("The state council announced on Monday that the new education policy "
            "will take effect across the region next year, and officials said it "
            "would give students in rural districts far better access to modern "
            "laboratories, libraries and digital learning resources than before. "
            "The measure was approved after months of public consultation and debate.")
    d5 = ("For more information contact the press office at press@example.com or call "
          "the newsroom directly. Heavy rain lashed the coastal city on Tuesday, and "
          "officials said the transport network was severely disrupted as commuters "
          "struggled through flooded streets during the long evening rush hour.")
    rows = [
        ("d1", good),
        ("d2", good + " A further update is expected shortly."),  # near-dup -> dropped
        ("d3", "ಕನ್ನಡ ಭಾಷೆ ಒಂದು ಸುಂದರ ಭಾಷೆ ಇದು ಬಹಳ ಹಳೆಯ ಶ್ರೀಮಂತ ಭಾಷೆ"),  # non-English -> dropped
        ("d4", "The note is short and it has very little content."),  # too short -> dropped
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
    assert "[EMAIL]" in d5 and "press@example.com" not in d5

    # files written and manifest valid
    assert (out / "manifest.json").exists()
    saved = json.loads((out / "stats.json").read_text())
    assert saved["final"]["tokens"] <= saved["baseline"]["tokens"]
    assert stats["manifest"]["license"] == "CommonCrawl-ToU"
