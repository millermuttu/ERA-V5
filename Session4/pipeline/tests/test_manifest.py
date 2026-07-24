from manifest import content_hash, build_manifest, validate_manifest


def test_content_hash_deterministic_and_order_sensitive():
    texts = ["ಒಂದು", "ಎರಡು", "ಮೂರು"]
    assert content_hash(texts) == content_hash(list(texts))
    assert content_hash(texts) != content_hash(["ಎರಡು", "ಒಂದು", "ಮೂರು"])


def test_valid_manifest_not_blocked():
    m = build_manifest("ai4bharat/sangraha::unverified/kan", "CC-BY-4.0",
                       "abc123", ["ಒಂದು", "ಎರಡು"], 42, "kan")
    v = validate_manifest(m)
    assert v["blocked"] is False and v["missing"] == []


def test_unknown_license_and_missing_field_block():
    m = build_manifest("s", "CC-BY-4.0", "h", ["ಒಂದು"], 1, "kan")
    bad_license = dict(m, license="unknown")
    assert validate_manifest(bad_license)["blocked"] is True
    missing = dict(m)
    del missing["content_hash"]
    assert validate_manifest(missing)["blocked"] is True
