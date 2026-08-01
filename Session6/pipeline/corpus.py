"""Deterministic toy multi-lane corpus generator (no network, seeded).

Produces documents across 5 training lanes plus a fixed set of
eval/benchmark candidate documents modeled on the firewall's gate shapes.
"""
import random

SEED = 6

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

# (benchmark_id, never_train, overlap_pct, canary_match, benchmark_derived)
EVAL_CANDIDATES = [
    ("mmlu-sample", True, 91.0, False, False),
    ("gsm8k-derived-blog", False, 12.0, False, True),
    ("held-out-eval-a", True, 5.0, True, False),
    ("benchmark-overlap-high", False, 61.0, False, False),
    ("clean-training-like", False, 3.0, False, False),
    ("canary-string-hit", False, 0.5, True, False),
    ("low-overlap-clean", False, 1.0, False, False),
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


def generate_corpus(n_per_lane=8):
    """Returns (training_docs, eval_docs), both lists of doc dicts."""
    rng = random.Random(SEED)
    docs = [_gen_doc(rng, lane, i) for lane in LANES for i in range(n_per_lane)]

    eval_docs = []
    vocab = _VOCAB_BY_LANE["general"]
    for i, (bench_id, never_train, overlap, canary, derived) in enumerate(EVAL_CANDIDATES):
        eval_docs.append({
            "doc_id": f"eval-doc-{i:03d}",
            "lane": "eval",
            "segments": [{"role": "response", "text": _lorem(rng, vocab, 30)}],
            "benchmark_id": bench_id,
            "never_train": never_train,
            "benchmark_overlap_pct": overlap,
            "canary_match": canary,
            "benchmark_derived": derived,
        })
    return docs, eval_docs


def doc_text(doc):
    return " ".join(seg["text"] for seg in doc["segments"])
