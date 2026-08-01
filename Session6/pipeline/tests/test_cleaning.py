from pipeline.cleaning import (CANARY, build_eval_fingerprint_set, has_canary, jaccard,
                                mask_structured, near_duplicates, overlap_pct, shingles)
from pipeline.corpus import doc_text, generate_corpus
from pipeline.train_loop import Config, build_world


# --- the vendored detectors themselves --------------------------------------

def test_overlap_pct_is_zero_for_unrelated_text():
    fps = build_eval_fingerprint_set(["the quick brown fox jumps over the lazy dog again today"])
    assert overlap_pct("completely unrelated words with nothing at all in common here", fps) == 0.0


def test_overlap_pct_is_total_for_a_verbatim_copy():
    text = "the quick brown fox jumps over the lazy dog again today please"
    assert overlap_pct(text, build_eval_fingerprint_set([text])) == 100.0


def test_overlap_pct_is_partial_for_a_spliced_span():
    evil = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    fps = build_eval_fingerprint_set([evil])
    pct = overlap_pct(evil + " " + " ".join(f"clean{i}" for i in range(20)), fps)
    assert 0.0 < pct < 100.0


def test_canary_detection():
    assert has_canary(f"some text {CANARY} more text")
    assert not has_canary("some text without it")


def test_mask_structured_masks_and_counts():
    masked, counts = mask_structured("mail a@b.com or call +91 9876543210 see http://x.y/z")
    assert counts == {"email": 1, "phone": 1, "url": 1, "ip": 0}
    assert "a@b.com" not in masked and "9876543210" not in masked


def test_near_duplicates_flags_a_near_copy_not_an_unrelated_doc():
    base = " ".join(f"word{i}" for i in range(40))
    near = " ".join(f"word{i}" for i in range(38)) + " today again"
    far = " ".join(f"other{i}" for i in range(40))
    dupes = near_duplicates([("base", base), ("near", near), ("far", far)])
    assert dupes == {"near": "base"}, dupes


def test_jaccard_bounds():
    a, b = shingles("one two three four five six"), shingles("one two three four five six")
    assert jaccard(a, b) == 1.0
    assert jaccard(a, shingles("totally different tokens entirely here now")) == 0.0


# --- the detectors must actually have something to detect -------------------

def test_corpus_contains_the_defects_the_cleaning_pass_looks_for():
    """Guard against silent regression to declared values: if the planted
    contamination disappears, every detector returns 0 and the manifest
    fields become measurements of nothing."""
    world = build_world(Config())
    shards = world["training_shards"]

    contaminated = [s for s in shards if s["benchmark_overlap_pct"] > 0]
    assert contaminated, "no training shard overlaps the eval set"
    assert any(s["benchmark_overlap_pct"] > 25.0 for s in contaminated), \
        "no training shard exceeds the firewall's overlap threshold"
    assert any(s["canary_match"] for s in shards), "no canary planted"
    assert any(s["pii_found"] for s in shards), "no PII planted"
    assert world["duplicates"], "no near-duplicates planted"


def test_measured_defects_actually_block_admission():
    world = build_world(Config())
    by_id = {m["shard_id"]: m for m in world["training_manifests"]}
    for shard in world["training_shards"]:
        manifest = by_id[shard["shard_id"]]
        if shard["benchmark_overlap_pct"] > 25.0 or shard["canary_match"]:
            assert manifest["admission"] == "blocked", shard["shard_id"]
    for dup_id in world["duplicates"]:
        assert by_id[dup_id]["admission"] == "blocked"
        assert by_id[dup_id]["dedup_status"].startswith("near_duplicate_of:")


def test_pii_never_reaches_a_token_id():
    """PII is masked before tokenization, so the raw identifiers must not
    survive anywhere in the tokenizer's vocabulary."""
    world = build_world(Config())
    vocab = set(world["tokenizer"].vocab)
    for leaked in ("priya.sharma", "9876543210", "254"):
        assert leaked not in vocab, leaked
    screened = [m for m in world["training_manifests"] if m["pii_screen_status"] == "masked"]
    assert screened, "no manifest recorded a PII hit"
    assert all(sum(m["pii_counts"].values()) > 0 for m in screened)


def test_no_manifest_field_is_a_blanket_constant():
    """dedup_status and pii_screen_status used to be default arguments, so
    every manifest carried an identical asserted value."""
    manifests = build_world(Config())["training_manifests"]
    assert len({m["dedup_status"] for m in manifests}) > 1
    assert len({m["pii_screen_status"] for m in manifests}) > 1
