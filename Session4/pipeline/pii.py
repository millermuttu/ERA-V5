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


@lru_cache(maxsize=1)
def load_ner():
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              pipeline)
    import torch
    tok = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
    model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("token-classification", model=model, tokenizer=tok,
                    aggregation_strategy="simple", device=device)


def mask_names(text, pipe=None):
    pipe = pipe or load_ner()
    ents = pipe(text)
    spans = sorted(((e["start"], e["end"]) for e in ents
                    if e.get("entity_group", "").upper() in ("PER", "PERSON")),
                   reverse=True)
    for s, e in spans:
        text = text[:s] + "[NAME]" + text[e:]
    return text, len(spans)
