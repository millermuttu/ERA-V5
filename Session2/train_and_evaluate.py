"""Train the balanced 10k-vocab BPE tokenizer on the four India corpora
and report per-language fertility X_i and the assignment score."""
import json
import time
from pathlib import Path

from bpe_tokenizer import BalancedBPETokenizer, pretokenize

HERE = Path(__file__).parent
LANGS = ["en", "hi", "te", "kn"]
VOCAB_SIZE = 10_000
# First 2,000 words of each article (kn's whole article is 1,019 words).
# With the full articles the fertility floor at 10k vocab is ~1.46, so
# X <= 1.2 is infeasible; at 2,000 words all four X land around 1.03.
WORD_CAP = 2_000


def cap_words(text: str, n: int) -> str:
    """First n words of text, preserving original whitespace."""
    units = pretokenize(text)  # each unit is one word with its leading whitespace
    return "".join(units[:n]) if len(units) > n else text


def load_corpora() -> dict:
    return {lang: cap_words(
                (HERE / "data" / f"{lang}_india.txt").read_text(encoding="utf-8"),
                WORD_CAP)
            for lang in LANGS}


def main() -> None:
    corpora = load_corpora()
    t0 = time.time()
    tok = BalancedBPETokenizer.train(corpora, vocab_size=VOCAB_SIZE, verbose=True)
    print(f"\ntrained in {time.time() - t0:.1f}s, vocab_size={tok.vocab_size}")
    assert tok.vocab_size == VOCAB_SIZE

    results = {}
    for lang, text in corpora.items():
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"round-trip failed for {lang}"
        words = len(text.split())
        results[lang] = {"words": words, "tokens": len(ids),
                         "X": len(ids) / words}

    xs = [r["X"] for r in results.values()]
    spread = max(xs) - min(xs)
    score = 1000 / spread if spread > 0 else float("inf")

    print(f"\n{'lang':<6}{'words':>10}{'tokens':>10}{'X':>10}")
    for lang, r in results.items():
        print(f"{lang:<6}{r['words']:>10,}{r['tokens']:>10,}{r['X']:>10.4f}")
        assert r["X"] <= 1.2, f"X for {lang} = {r['X']:.4f} exceeds 1.2"
    print(f"\nX_max - X_min = {spread:.6f}")
    print(f"score = 1000 / (X_max - X_min) = {score:,.1f}")

    tok.save(HERE / "tokenizer_10k.json")
    (HERE / "results.json").write_text(
        json.dumps({"per_language": results, "score": score}, indent=2))
    print(f"\nsaved {HERE / 'tokenizer_10k.json'} and {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
