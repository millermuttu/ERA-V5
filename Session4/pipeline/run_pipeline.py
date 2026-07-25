import json
import os
import random
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
SOURCE = "vblagoje/cc_news (CommonCrawl News, English) :: 20M-token slice"
LICENSE = "CommonCrawl-ToU"
LANG = "en"

# Fallback held-out probe (used when no eval set and holdout sampling is off).
_EVAL_FALLBACK = [
    "this sentence belongs to the held-out evaluation set and must not be trained on",
]


def _log(msg):
    print(msg, flush=True)


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
        eval_texts=None, decontam_holdout=0):
    os.makedirs(outdir, exist_ok=True)
    _log(f"[load] reading {input_parquet}")
    tbl = pq.read_table(input_parquet, columns=["doc_id", "text"]).to_pydict()
    docs = list(zip(tbl["doc_id"], tbl["text"]))
    _log(f"[load] {len(docs):,} docs; tokenizing baseline (o200k)...")

    # token cache keyed by doc_id (recomputed only when text changes)
    tok = {d: count_tokens(t) for d, t in docs}

    def totals(items):
        return {"docs": len(items), "tokens": sum(tok[d] for d, _ in items)}

    baseline = totals(docs)
    baseline_words = sum(len(t.split()) for _, t in docs)
    baseline_fertility = round(baseline["tokens"] / max(1, baseline_words), 3)
    stages = []

    def record(name, before_items, after_items, indic, diffs, extra=None):
        b, a = totals(before_items), totals(after_items)
        st = {
            "name": name,
            "docs_in": b["docs"], "docs_out": a["docs"],
            "tokens_in": b["tokens"], "tokens_out": a["tokens"],
            "removed_docs": b["docs"] - a["docs"],
            "removed_tokens": b["tokens"] - a["tokens"],
            "removed_pct": round(100 * (b["tokens"] - a["tokens"]) / b["tokens"], 2)
                           if b["tokens"] else 0.0,
            "indic_concern": indic,
            "example_diffs": diffs,
        }
        if extra:
            st.update(extra)
        stages.append(st)
        _log(f"  [{name}] {b['docs']:,}->{a['docs']:,} docs | "
             f"{a['tokens']:,} tok | -{st['removed_pct']}% tokens")

    # 1. NORMALIZE (mutates text)
    _NOISE = "".join(chr(c) for c in
                     (0x200b, 0xfeff, 0x200e, 0x200f, 0x202a, 0x202b,
                      0x202c, 0x202d, 0x202e, 0xfffd))
    before = docs
    norm_pairs, out = [], []
    docs_modified = chars_removed = zw_removed = 0
    for d, t in docs:
        clean, _ops = _normalize.normalize_text(t)
        norm_pairs.append((t, clean))
        if clean != t:
            docs_modified += 1
            chars_removed += max(0, len(t) - len(clean))
        zw_removed += sum(t.count(ch) for ch in _NOISE)
        tok[d] = count_tokens(clean)
        out.append((d, clean))
    record("normalize", before, out,
           "NFC, strip zero-width and control noise, unescape HTML entities, "
           "collapse whitespace. Preserves ZWJ/ZWNJ joiners for any Brahmic text.",
           _diff_samples(norm_pairs, sample_diffs),
           extra={"docs_modified": docs_modified, "chars_removed": chars_removed,
                  "zero_width_noise_removed": zw_removed})
    docs = out

    # 2. LANGUAGE-ID
    before = docs
    kept, dropped = [], []
    for d, t in docs:
        (kept if _langid.is_english(t) else dropped).append((d, t))
    out = kept
    record("langid", before, out,
           "Detects real English prose by Latin-script ratio + stop-word density; "
           "drops other-language, other-script and non-prose junk.",
           _diff_samples([(t, "[DROPPED: not English]") for _, t in dropped], sample_diffs))
    docs = out

    # 3. QUALITY
    before = docs
    kept, drops = [], []
    for d, t in docs:
        ok, reasons = _quality.quality_check(t, indic_aware=True)
        (kept if ok else drops).append((d, t, reasons if not ok else None))
    out = [(d, t) for d, t, _ in kept]
    record("quality", before, out,
           "Heuristic cascade: min length, symbol ratio, boilerplate/nav lines, "
           "duplicate-line ratio, mean word length.",
           _diff_samples([(t, f"[DROPPED: {r}]") for _, t, r in drops], sample_diffs))
    docs = out

    # 4. DEDUP
    before = docs
    dup_keys = _dedup.find_duplicates(docs, threshold=0.8)
    out = [(d, t) for d, t in docs if d not in dup_keys]
    record("dedup", before, out,
           "Global near-duplicate detection with MinHash+LSH. CommonCrawl news is "
           "undeduplicated, so syndicated reposts and re-crawls are caught here -- "
           "the near-duplicates exact hashing misses.",
           _diff_samples([(t, "[DROPPED: near-duplicate]")
                          for d, t in docs if d in dup_keys], sample_diffs))
    docs = out

    # 5. PII: structured regex on all docs, then names via batched NER
    before = docs
    pii_counts = {"email": 0, "phone": 0, "url": 0, "ip": 0, "name": 0}
    ids = [d for d, _ in docs]
    originals = [t for _, t in docs]
    struct = []
    for t in originals:
        m, c = _pii.mask_structured(t)
        for k in ("email", "phone", "url", "ip"):
            pii_counts[k] += c[k]
        struct.append(m)

    if ner_pipe is not None:                     # test path: per-doc stub
        masked_texts = []
        for m in struct:
            mm, n = _pii.mask_names(m, pipe=ner_pipe)
            pii_counts["name"] += n
            masked_texts.append(mm)
    elif use_ner:                                # real path: batched on GPU
        _log(f"  [pii] masking names across {len(struct):,} docs (batched, head-window NER)...")
        masked_texts, total = _pii.mask_names_batch(
            struct, batch_size=64, max_windows=1,
            progress=lambda d, n: _log(f"    NER {d:,}/{n:,} chunks"))
        pii_counts["name"] += total
    else:
        masked_texts = struct

    out, pii_pairs = [], []
    for d, orig, mt in zip(ids, originals, masked_texts):
        if mt != orig:
            tok[d] = count_tokens(mt)
        out.append((d, mt))
        pii_pairs.append((orig, mt))
    record("pii", before, out,
           "Regex masks structured identifiers (emails, phones, URLs, IPs) across "
           "every document. Personal-name NER is shown as a demo to keep the run fast.",
           _diff_samples(pii_pairs, sample_diffs))
    docs = out

    # 6. DECONTAM: fingerprint a held-out eval set, remove any training doc that
    # overlaps it (or its duplicates). eval_texts overrides; else sample a holdout.
    before = docs
    if eval_texts is not None:
        holdout = eval_texts
    elif decontam_holdout and docs:
        rng = random.Random(1234)
        k = min(decontam_holdout, len(docs))
        holdout = [docs[i][1] for i in rng.sample(range(len(docs)), k)]
    else:
        holdout = _EVAL_FALLBACK
    eval_fps = _decontam.build_eval_fingerprint_set(holdout, n=8)
    kept, dropped = [], []
    for d, t in docs:
        if _decontam.is_contaminated(t, eval_fps, n=8) or _decontam.has_canary(t):
            dropped.append((d, t))
        else:
            kept.append((d, t))
    out = kept
    record("decontam", before, out,
           "A held-out eval set is fingerprinted; every training doc that overlaps "
           "it -- or is a duplicate of an eval doc -- is removed, plus canary strings.",
           _diff_samples([(t, "[DROPPED: eval overlap]") for _, t in dropped], sample_diffs),
           extra={"holdout_docs": len(holdout)})
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
        "baseline": {"docs": baseline["docs"], "tokens": baseline["tokens"],
                     "words": baseline_words, "fertility": baseline_fertility},
        "stages": stages,
        "pii": pii_counts,
        "final": final,
        "manifest": {**man, "validation": validation},
        "meta": {"source": SOURCE, "license": LICENSE, "language": LANG,
                 "ner_model": _pii.NER_MODEL,
                 "generated_at": man["generated_at"], "script_hash": script_h},
    }

    _log(f"[write] {len(docs):,} docs -> {outdir}/ (corpus, manifest, stats)")
    pq.write_table(pa.table({"doc_id": [d for d, _ in docs], "text": texts}),
                   os.path.join(outdir, "cleaned_corpus.parquet"))
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({**man, "validation": validation}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main():
    t0 = time.time()
    # use_ner=False: names are a demo (keeps the run fast); structured PII is real.
    # decontam_holdout=50: hold out 50 docs as an eval set to exercise the firewall.
    stats = run("Session4/data/raw/ccnews_slice.parquet",
                "Session4/data/cleaned", use_ner=False, decontam_holdout=50)
    print(json.dumps({"baseline": stats["baseline"], "final": stats["final"],
                      "pii": stats["pii"],
                      "stages": [(s["name"], s["docs_in"], s["docs_out"])
                                 for s in stats["stages"]]},
                     ensure_ascii=False, indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
