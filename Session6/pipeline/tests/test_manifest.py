from pipeline.corpus import generate_corpus
from pipeline.manifest import (ADMIT_THRESHOLD, HARD_BLOCK_CAP, build_manifest,
                                build_manifests, decide_admission, score_manifest)
from pipeline.shards import build_shard, build_shards
from pipeline.tokenizer import build_tokenizer


def _tokenizer():
    docs, eval_docs = generate_corpus(n_per_lane=2)
    from pipeline.corpus import doc_text
    texts = [doc_text(d) for d in docs + eval_docs]
    return build_tokenizer(texts)


def test_complete_high_scoring_shard_is_admitted():
    tok = _tokenizer()
    docs, _ = generate_corpus(n_per_lane=2)
    shard = build_shard(docs[0], tok)
    manifest = build_manifest(shard, license_tier="safe")
    assert manifest["admission"] == "admitted"
    assert manifest["admission_score"] > ADMIT_THRESHOLD


def test_missing_hard_required_field_is_not_admitted():
    manifest = {"tokenizer_hash": "", "cleaning_pipeline_hash": "abc",
                "eval_overlap_status": "clear", "license_tier": "safe"}
    manifest["admission_score"] = score_manifest(manifest)
    manifest["admission"] = decide_admission(manifest)
    assert manifest["admission"] in ("blocked", "held_for_review")
    assert manifest["admission_score"] <= HARD_BLOCK_CAP


def test_unsafe_license_is_blocked():
    manifest = {"tokenizer_hash": "abc", "cleaning_pipeline_hash": "def",
                "eval_overlap_status": "clear", "license_tier": "unsafe"}
    manifest["admission_score"] = score_manifest(manifest)
    manifest["admission"] = decide_admission(manifest)
    assert manifest["admission"] == "blocked"


def test_retokenizing_identical_docs_is_idempotent():
    tok = _tokenizer()
    docs, _ = generate_corpus(n_per_lane=2)
    shard_a = build_shard(docs[0], tok)
    shard_b = build_shard(docs[0], tok)
    assert shard_a["shard_id"] == shard_b["shard_id"]
    assert shard_a["content_hash"] == shard_b["content_hash"]


def test_changed_document_changes_hash():
    tok = _tokenizer()
    docs, _ = generate_corpus(n_per_lane=2)
    shard_a = build_shard(docs[0], tok)
    mutated = dict(docs[0])
    mutated["segments"] = [{"role": "response", "text": "completely different words entirely"}]
    shard_b = build_shard(mutated, tok)
    assert shard_a["content_hash"] != shard_b["content_hash"]


def test_firewall_tripped_shard_is_blocked_not_admitted():
    """A blocked shard used to score 100 and come out `admitted`; nothing
    leaked only because the training loop happened not to select it."""
    tok = _tokenizer()
    _, eval_docs = generate_corpus(n_per_lane=2)
    blocked_doc = next(d for d in eval_docs if d["never_train"])
    manifest = build_manifest(build_shard(blocked_doc, tok), license_tier="safe")
    assert manifest["eval_overlap_status"] == "blocked_or_unknown"
    assert manifest["admission"] == "blocked"
    assert manifest["admission_score"] <= HARD_BLOCK_CAP
    assert "never_train" in manifest["eval_firewall_reasons"]


def test_every_firewall_flagged_eval_shard_is_blocked():
    tok = _tokenizer()
    _, eval_docs = generate_corpus(n_per_lane=2)
    shards = build_shards(eval_docs, tok)
    manifests = build_manifests(shards, license_tier="safe")
    for shard, manifest in zip(shards, manifests):
        tripped = (shard["never_train"] or shard["benchmark_overlap_pct"] > 25
                   or shard["canary_match"] or shard["benchmark_derived"])
        assert (manifest["admission"] == "blocked") == bool(tripped), shard["shard_id"]


def test_eval_candidate_shard_gets_blocked_overlap_status():
    tok = _tokenizer()
    _, eval_docs = generate_corpus(n_per_lane=2)
    shards = build_shards(eval_docs, tok)
    manifests = build_manifests(shards, license_tier="safe")
    flagged = [m for m, s in zip(manifests, shards)
               if s["never_train"] or s["benchmark_overlap_pct"] > 25 or s["canary_match"] or s["benchmark_derived"]]
    assert all(m["eval_overlap_status"] == "blocked_or_unknown" for m in flagged)
