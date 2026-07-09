"""Download the Wikipedia 'India' article in en, hi, te, kn as plain text.

Resolves the article title on each wiki by following the English article's
language links, then fetches plain-text extracts via the MediaWiki API.
Fails loudly on any missing link or empty extract.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "ERA-Session2-BPE/1.0 (educational assignment)"
DATA_DIR = Path(__file__).parent / "data"
TARGET_LANGS = ["hi", "te", "kn"]


def api_get(host: str, params: dict) -> dict:
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def get_titles() -> dict:
    """Resolve each wiki's title for the India article via language links."""
    data = api_get("en.wikipedia.org", {
        "action": "query", "titles": "India", "prop": "langlinks",
        "lllimit": "500", "format": "json", "redirects": "1",
    })
    page = next(iter(data["query"]["pages"].values()))
    titles = {"en": "India"}
    for link in page.get("langlinks", []):
        if link["lang"] in TARGET_LANGS:
            titles[link["lang"]] = link["*"]
    missing = set(["en"] + TARGET_LANGS) - set(titles)
    if missing:
        sys.exit(f"No language link found for: {sorted(missing)}")
    return titles


def fetch_extract(lang: str, title: str) -> str:
    data = api_get(f"{lang}.wikipedia.org", {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "titles": title, "format": "json", "redirects": "1",
    })
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    if not text.strip():
        sys.exit(f"Empty extract for {lang}:{title}")
    return text


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for lang, title in get_titles().items():
        text = fetch_extract(lang, title)
        out = DATA_DIR / f"{lang}_india.txt"
        out.write_text(text, encoding="utf-8")
        print(f"{lang}: {title!r} -> {out.name}: "
              f"{len(text):,} chars, {len(text.split()):,} words")


if __name__ == "__main__":
    main()
