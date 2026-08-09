# ERA V5

Coursework repo. One directory per session (`Session1/` … `SessionN/`), each holding the
scraped course material plus whatever the assignment asked for.

## Layout convention

```
SessionN/
  reference/
    sessionN-lesson.md          # the lesson prose, transcribed
    widgets/widget_K_name.html  # each interactive widget, saved verbatim
    widgets-scripts-dump.txt    # inline <script> bodies, concatenated for grepping
  README.md | index.html | submission.md   # the deliverable
```

`openspec/` holds spec-driven change proposals for the larger sessions.

## Scraping a new session

The lesson page is behind the Axiom login, so `WebFetch`/`curl` on it returns the sign-in
shell. The widgets are **not** — they are public static files. Procedure:

1. **Connect Chrome.** `list_connected_browsers` → `select_browser <deviceId>`. If the list is
   empty, ask the user to open Chrome with the Claude extension, logged into the same account.
   Nothing below works without this, and there is no unauthenticated fallback for the prose.

2. **Get the prose.** `navigate` to the lesson URL, then `get_page_text`. Transcribe it into
   `reference/sessionN-lesson.md` — keep the section numbering and headings, convert the widget
   paragraphs to `*Widget: …*` italic notes, keep every number and formula verbatim. Quote the
   assignment section as a blockquote; it is the spec for the deliverable.

3. **List the widgets.** `javascript_tool`:
   ```js
   [...document.querySelectorAll('iframe')].map(f => f.src).join('\n')
   ```
   Widget srcs look like `https://axiom.theschoolofai.in/widgets/sN_widget_K_name.html`.
   YouTube iframes come back in the same list — record those URLs in the lesson header and
   skip them. The list is duplicated (the page renders the lesson twice); dedupe.

4. **Download them.** They are public, so plain `curl` works and the browser is not needed:
   ```bash
   curl -sfL -o "reference/widgets/widget_K_name.html" \
        "https://axiom.theschoolofai.in/widgets/sN_widget_K_name.html"
   ```
   Save as `widget_K_name.html` (drop the `sN_` prefix — the session dir already says which).

5. **Dump the scripts.** Strip `<style>`, concatenate every inline `<script>` body with an
   `=== filename ===` banner, write `reference/widgets-scripts-dump.txt`. This is what you
   grep; the widget HTML itself is mostly CSS.

6. **Mine the numbers.** Read the scripts and write `reference/mined-numbers.md`. What matters:
   slider `min`/`max`/`value` defaults, `<option>` values, hard-coded data tables (language
   shares, model-scale lists, demo words), the formulas the widget computes, and any pass/flag/
   fail gate thresholds. These are the baselines an implementation gets measured against, and
   the widget source is the only place several of them are written down.

7. **Close the tab** (`tabs_close_mcp`) when done.

## Working style

Commit messages use the `[ERA-V5][muttu]:` prefix — use the `era-commit` skill.
Work happens on `master`. Do not push unless asked.
