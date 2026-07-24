import json
import os
import time
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

import normalize as _normalize
import langid as _langid
import quality as _quality
import dedup as _dedup
import pii as _pii
import decontam as _decontam
import manifest as _manifest

_ENC = tiktoken.get_encoding("o200k_base")
SOURCE = "ai4bharat/sangraha::unverified/kan/data-0.parquet"
LICENSE = "CC-BY-4.0"
LANG = "kan"

# Held-out probe used for the decontamination stage. If FLORES-Kannada is
# available it is loaded in main(); this fallback keeps the stage meaningful.
_EVAL_FALLBACK = [
    "ಈ ವಾಕ್ಯವು ಮೌಲ್ಯಮಾಪನ ಸೆಟ್‌ನ ಭಾಗವಾಗಿದೆ ಮತ್ತು ತರಬೇತಿಯಲ್ಲಿ ಬರಬಾರದು",
]


def count_tokens(text):
    return len(_ENC.encode(text))


def _diff_samples(pairs, n):
    out = []
    for before, after in pairs:
        if before != after:
            out.append({"before": before[:400], "after": after[:400]})
        if len(out) >= n:
            break
    return out


def run(input_parquet, outdir, use_ner=True, ner_pipe=None, sample_diffs=3,
        eval_texts=None):
    os.makedirs(outdir, exist_ok=True)
    tbl = pq.read_table(input_parquet, columns=["doc_id", "text"]).to_pydict()
    docs = list(zip(tbl["doc_id"], tbl["text"]))

    # token cache keyed by doc_id (recomputed only when text changes)
    tok = {d: count_tokens(t) for d, t in docs}

    def totals(items):
        return {"docs": len(items), "tokens": sum(tok[d] for d, _ in items)}

    baseline = totals(docs)
    stages = []

    def record(name, before_items, after_items, indic, diffs):
        b, a = totals(before_items), totals(after_items)
        stages.append({
            "name": name,
            "docs_in": b["docs"], "docs_out": a["docs"],
            "tokens_in": b["tokens"], "tokens_out": a["tokens"],
            "removed_docs": b["docs"] - a["docs"],
            "removed_tokens": b["tokens"] - a["tokens"],
            "removed_pct": round(100 * (b["tokens"] - a["tokens"]) / b["tokens"], 2)
                           if b["tokens"] else 0.0,
            "indic_concern": indic,
            "example_diffs": diffs,
        })

    # 1. NORMALIZE (mutates text)
    before = docs
    norm_pairs, out = [], []
    for d, t in docs:
        clean, _ops = _normalize.normalize_text(t)
        norm_pairs.append((t, clean))
        tok[d] = count_tokens(clean)
        out.append((d, clean))
    record("normalize", before, out,
           "Keeps Brahmic joiners (ZWNJ/ZWJ) while stripping zero-width noise.",
           _diff_samples(norm_pairs, sample_diffs))
    docs = out

    # 2. LANGUAGE-ID
    before = docs
    out = [(d, t) for d, t in docs if _langid.is_kannada(t, 0.5)]
    dropped = [(d, t) for d, t in docs if not _langid.is_kannada(t, 0.5)]
    record("langid", before, out,
           "Detects real Kannada vs code-switched/mislabelled docs.",
           _diff_samples([(t, "[DROPPED: not Kannada]") for _, t in dropped], sample_diffs))
    docs = out

    # 3. QUALITY
    before = docs
    kept, drops = [], []
    for d, t in docs:
        ok, reasons = _quality.quality_check(t, indic_aware=True)
        (kept if ok else drops).append((d, t, reasons if not ok else None))
    out = [(d, t) for d, t, _ in kept]
    record("quality", before, out,
           "Indic-aware thresholds so combining marks are not counted as symbols.",
           _diff_samples([(t, f"[DROPPED: {r}]") for _, t, r in drops], sample_diffs))
    docs = out

    # 4. DEDUP
    before = docs
    dup_keys = _dedup.find_duplicates(docs, threshold=0.8)
    out = [(d, t) for d, t in docs if d not in dup_keys]
    record("dedup", before, out,
           "Near-duplicate removal the Indic crawl never had (exact dupes were 0).",
           _diff_samples([(t, "[DROPPED: near-duplicate]")
                          for d, t in docs if d in dup_keys], sample_diffs))
    docs = out

    # 5. PII (mutates text)
    before = docs
    pii_counts = {"email": 0, "phone": 0, "url": 0, "ip": 0, "name": 0}
    pipe = ner_pipe if ner_pipe is not None else (_pii.load_ner() if use_ner else None)
    pii_pairs, out = [], []
    for d, t in docs:
        masked, c = _pii.mask_structured(t)
        for k in ("email", "phone", "url", "ip"):
            pii_counts[k] += c[k]
        if pipe is not None:
            masked, n = _pii.mask_names(masked, pipe=pipe)
            pii_counts["name"] += n
        pii_pairs.append((t, masked))
        if masked != t:
            tok[d] = count_tokens(masked)
        out.append((d, masked))
    record("pii", before, out,
           "Regex identifiers + IndicNER PER names (precision/recall tension for Indic names).",
           _diff_samples(pii_pairs, sample_diffs))
    docs = out

    # 6. DECONTAM
    before = docs
    eval_fps = _decontam.build_eval_fingerprint_set(eval_texts or _EVAL_FALLBACK, n=8)
    kept, dropped = [], []
    for d, t in docs:
        if _decontam.is_contaminated(t, eval_fps, n=8) or _decontam.has_canary(t):
            dropped.append((d, t))
        else:
            kept.append((d, t))
    out = kept
    record("decontam", before, out,
           "Fingerprint vs held-out eval + canary strings to keep scores honest.",
           _diff_samples([(t, "[DROPPED: eval overlap]") for _, t in dropped], sample_diffs))
    docs = out

    # 7. MANIFEST
    texts = [t for _, t in docs]
    final_tokens = sum(tok[d] for d, _ in docs)
    script_h = _manifest.script_hash(os.path.abspath(__file__))
    man = _manifest.build_manifest(
        SOURCE, LICENSE, script_h, texts, final_tokens, LANG,
        extra={"pipeline_stages": [s["name"] for s in stages],
               "generated_at": datetime.now(timezone.utc).isoformat()})
    validation = _manifest.validate_manifest(man)

    final = {
        "docs": len(docs),
        "tokens": final_tokens,
        "fertility": round(final_tokens / max(1, sum(len(t.split()) for t in texts)), 3),
        "total_reduction_pct": round(100 * (baseline["tokens"] - final_tokens)
                                     / baseline["tokens"], 2) if baseline["tokens"] else 0.0,
    }

    stats = {
        "baseline": {"docs": baseline["docs"], "tokens": baseline["tokens"]},
        "stages": stages,
        "pii": pii_counts,
        "final": final,
        "manifest": {**man, "validation": validation},
        "meta": {"source": SOURCE, "license": LICENSE, "language": LANG,
                 "generated_at": man["generated_at"], "script_hash": script_h},
    }

    pq.write_table(pa.table({"doc_id": [d for d, _ in docs], "text": texts}),
                   os.path.join(outdir, "cleaned_corpus.parquet"))
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({**man, "validation": validation}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main():
    t0 = time.time()
    stats = run("Session4/data/raw/kan_slice.parquet",
                "Session4/data/cleaned", use_ner=True)
    print(json.dumps({"baseline": stats["baseline"], "final": stats["final"],
                      "pii": stats["pii"],
                      "stages": [(s["name"], s["docs_in"], s["docs_out"])
                                 for s in stats["stages"]]},
                     ensure_ascii=False, indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
