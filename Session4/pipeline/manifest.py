import hashlib

REQUIRED = ["source", "license", "cleaning_script_hash", "content_hash",
            "token_count", "doc_count", "language"]
_UNSAFE = {"", "unknown", "unsafe", "none", "proprietary"}


def content_hash(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def script_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_manifest(source, license, cleaning_script_hash, texts, token_count,
                   language, extra=None):
    m = {
        "source": source,
        "license": license,
        "cleaning_script_hash": cleaning_script_hash,
        "content_hash": content_hash(texts),
        "token_count": token_count,
        "doc_count": len(texts) if hasattr(texts, "__len__") else None,
        "language": language,
    }
    if extra:
        m.update(extra)
    return m


def validate_manifest(m):
    missing = [k for k in REQUIRED if m.get(k) in (None, "", 0)]
    unsafe = str(m.get("license", "")).strip().lower() in _UNSAFE
    return {"blocked": bool(missing) or unsafe,
            "missing": missing, "unsafe_license": unsafe}
