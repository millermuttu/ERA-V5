import re
from functools import lru_cache

_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")


def mask_structured(text):
    counts = {"email": 0, "phone": 0, "url": 0, "ip": 0}
    for pat, tag, key in [(_URL, "URL", "url"), (_EMAIL, "EMAIL", "email"),
                          (_IP, "IP", "ip"), (_PHONE, "PHONE", "phone")]:
        text, n = pat.subn(f"[{tag}]", text)
        counts[key] += n
    return text, counts


_NER_WINDOW = 800   # chars per NER window (~<512 wordpieces for Kannada)
# Non-gated, distilled multilingual NER. Trained on high-resource languages, so
# recall on Kannada names is limited -- we report the real (conservative) count
# honestly rather than pretend it is exhaustive.
NER_MODEL = "Davlan/distilbert-base-multilingual-cased-ner-hrl"


def _chunks_with_starts(text, window=_NER_WINDOW, max_chunks=None):
    """Split text into <=window-char windows on word boundaries.

    Returns (chunks, starts) where starts[i] is chunk i's offset in `text`,
    so entity offsets can be remapped to full-document coordinates.
    max_chunks caps how many leading windows are produced (head-only scan).
    """
    chunks, starts = [], []
    start, n = 0, len(text)
    while start < n:
        end = min(n, start + window)
        if end < n:
            sp = text.rfind(" ", start + 1, end)
            if sp != -1:
                end = sp
        chunks.append(text[start:end])
        starts.append(start)
        start = end
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks, starts


@lru_cache(maxsize=1)
def load_ner_pipeline():
    """The raw transformers token-classification pipeline (cached)."""
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              pipeline)
    import torch
    tok = AutoTokenizer.from_pretrained(NER_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("token-classification", model=model, tokenizer=tok,
                    aggregation_strategy="simple", device=device)


def load_ner():
    """Per-document callable: chunk a doc and return remapped entity spans."""
    nlp = load_ner_pipeline()

    def _run(text):
        chunks, starts = _chunks_with_starts(text)
        if not chunks:
            return []
        out = []
        for ents, st in zip(nlp(chunks, batch_size=8), starts):
            for e in ents:
                out.append({**e, "start": e["start"] + st, "end": e["end"] + st})
        return out

    return _run


def _is_per(ent):
    return ent.get("entity_group", "").upper() in ("PER", "PERSON")


def mask_names(text, pipe=None):
    pipe = pipe or load_ner()
    spans = sorted(((e["start"], e["end"]) for e in pipe(text) if _is_per(e)),
                   reverse=True)
    for s, e in spans:
        text = text[:s] + "[NAME]" + text[e:]
    return text, len(spans)


def mask_names_batch(texts, nlp=None, batch_size=32, max_windows=None, progress=None):
    """Mask PER names across many docs, batching every chunk through the GPU
    in one streaming pass instead of one pipeline call per document.

    max_windows caps the leading windows scanned per doc (head-only NER);
    structured identifiers are still masked full-document by mask_structured.

    Returns (masked_texts, total_names).
    """
    nlp = nlp or load_ner_pipeline()

    # 1. flatten every doc into chunks, remembering which doc each came from.
    all_chunks, owner, offset = [], [], []
    for i, t in enumerate(texts):
        chunks, starts = _chunks_with_starts(t, max_chunks=max_windows)
        for c, s in zip(chunks, starts):
            all_chunks.append(c)
            owner.append(i)
            offset.append(s)

    spans = [[] for _ in texts]
    if all_chunks:
        done = 0
        # nlp accepts a list and batches internally; iterate to report progress.
        for (i, s), ents in zip(zip(owner, offset),
                                nlp(all_chunks, batch_size=batch_size)):
            for e in ents:
                if _is_per(e):
                    spans[i].append((e["start"] + s, e["end"] + s))
            done += 1
            if progress and done % 5000 == 0:
                progress(done, len(all_chunks))

    masked, total = [], 0
    for t, sp in zip(texts, spans):
        for a, b in sorted(sp, reverse=True):
            t = t[:a] + "[NAME]" + t[b:]
        total += len(sp)
        masked.append(t)
    return masked, total
