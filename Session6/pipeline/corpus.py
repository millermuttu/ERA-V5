"""Deterministic toy multi-lane corpus generator (no network, seeded).

Produces documents across 5 training lanes plus a fixed set of
eval/benchmark candidate documents modeled on the firewall's gate shapes.

The corpus deliberately plants real defects for the cleaning pass to find:
verbatim spans copied out of eval documents, a canary string, structured
PII, and near-duplicate documents. Without them the detectors in
`cleaning.py` would return 0 for everything and the firewall, PII and dedup
statuses would be measurements of nothing - which is exactly why those
fields used to be hardcoded.
"""
import random

from pipeline.cleaning import CANARY

SEED = 6

# Training docs that get a verbatim span spliced in from eval doc N, long
# enough to push measured n-gram overlap past the firewall's 25% threshold.
CONTAMINATED = {"general-doc-002": 0, "reasoning-doc-005": 3, "code-doc-007": 1}
CANARY_DOCS = {"general-doc-004"}
PII_DOCS = {"general-doc-006": "contact priya.sharma@example.com or +91 9876543210",
            "code-doc-003": "see http://internal.example.com/spec or 10.1.2.254"}
# Near-duplicates: value is the doc whose text gets copied (then lightly edited).
NEAR_DUPLICATES = {"general-doc-009": "general-doc-000", "indic-doc-008": "indic-doc-001"}

LANES = ["general", "code", "indic", "reasoning", "agentic"]

_VOCAB_BY_LANE = {
    "general": ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                "market", "report", "weather", "today"],
    "code": ["def", "return", "import", "class", "self", "for", "in", "range",
             "print", "if", "else", "while"],
    "indic": ["namaste", "dhanyavaad", "bharath", "yojana", "gram", "panchayat",
              "vikas", "seva", "shiksha", "arogya"],
    "reasoning": ["therefore", "because", "hypothesis", "proof", "assume",
                  "contradiction", "theorem", "conclude", "step", "given"],
    "agentic": ["user", "observe", "plan", "call_tool", "result", "respond",
                "goal", "action", "state", "next"],
}

AGENTIC_ROLES = ["user", "observation", "plan", "tool_call", "response"]
LOSS_BEARING_ROLES = {"plan", "tool_call", "response"}

# (benchmark_id, never_train, benchmark_derived) - declared provenance only.
# Overlap percentage and canary hits are now measured by `cleaning.py`, not
# asserted here.
EVAL_CANDIDATES = [
    ("mmlu-sample", True, False),
    ("gsm8k-derived-blog", False, True),
    ("held-out-eval-a", True, False),
    ("benchmark-overlap-high", False, False),
    ("clean-training-like", False, False),
    ("canary-string-hit", False, False),
    ("low-overlap-clean", False, False),
]


def _lorem(rng, vocab, n_words):
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def _gen_doc(rng, lane, idx, n_words=40):
    vocab = _VOCAB_BY_LANE[lane]
    doc_id = f"{lane}-doc-{idx:03d}"
    if lane == "agentic":
        segments = [
            {"role": AGENTIC_ROLES[step % len(AGENTIC_ROLES)], "text": _lorem(rng, vocab, 8)}
            for step in range(rng.randint(3, 5))
        ]
        return {"doc_id": doc_id, "lane": lane, "segments": segments}
    return {"doc_id": doc_id, "lane": lane,
            "segments": [{"role": "response", "text": _lorem(rng, vocab, n_words)}]}


def _set_text(doc, text):
    doc["segments"] = [dict(doc["segments"][0], text=text)]


def generate_corpus(n_per_lane=8):
    """Returns (training_docs, eval_docs), both lists of doc dicts.

    Eval documents are built first so training documents can copy verbatim
    spans out of them - the contamination the decontam pass then measures.
    """
    rng = random.Random(SEED)

    eval_docs = []
    vocab = _VOCAB_BY_LANE["general"]
    for i, (bench_id, never_train, derived) in enumerate(EVAL_CANDIDATES):
        eval_docs.append({
            "doc_id": f"eval-doc-{i:03d}",
            "lane": "eval",
            "segments": [{"role": "response", "text": _lorem(rng, vocab, 30)}],
            "benchmark_id": bench_id,
            # Provenance, not something detectable from the text itself: these
            # come from the source contract (Session 3) and stay declared.
            # `benchmark_overlap_pct` and `canary_match` are measured instead.
            "never_train": never_train,
            "benchmark_derived": derived,
        })

    docs = [_gen_doc(rng, lane, i) for lane in LANES for i in range(n_per_lane)]
    by_id = {d["doc_id"]: d for d in docs}

    for doc_id, eval_idx in CONTAMINATED.items():
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        stolen = " ".join(doc_text(eval_docs[eval_idx]).split()[:22])
        _set_text(doc, f"{stolen} {doc['segments'][0]['text']}")

    for doc_id in CANARY_DOCS:
        if doc_id in by_id:
            _set_text(by_id[doc_id], f"{by_id[doc_id]['segments'][0]['text']} {CANARY}")

    for doc_id, pii_text in PII_DOCS.items():
        if doc_id in by_id:
            _set_text(by_id[doc_id], f"{by_id[doc_id]['segments'][0]['text']} {pii_text}")

    for dup_id, source_id in NEAR_DUPLICATES.items():
        if dup_id in by_id and source_id in by_id:
            words = by_id[source_id]["segments"][0]["text"].split()
            _set_text(by_id[dup_id], " ".join(words[:-2] + ["today", "again"]))

    return docs, eval_docs


def doc_text(doc):
    return " ".join(seg["text"] for seg in doc["segments"])
