# ERA V5

Coursework repo. One directory per session (`Session1/` … `SessionN/`), each holding the
scraped course material plus whatever the assignment asked for.

## Layout convention

```
SessionN/
  reference/
    sessionN-lesson.md          # the lesson prose, transcribed
    widgets/widget_K_name.html  # each interactive widget, saved verbatim
    mined-numbers.md            # the defaults, thresholds and data tables read out of the widgets
  README.md | index.html | submission.md   # the deliverable
```

Only the lesson and the widgets are in every session; `mined-numbers.md` appears where the widgets
carried numbers worth extracting (Sessions 5 and 6).

The scrape also produces `reference/widgets-scripts-dump.txt` — every widget's inline `<script>`
concatenated into one grep target. That one is **working material, not a kept artifact**: it exists so
the numbers can be mined out of it while the deliverable is being built, and it is deleted once they
have been. Nothing is lost when it goes, because each script is still in its own file under
`widgets/`. Only Session 6 still carries one; Session 8's was removed when its cards were done. Its
absence in a session is not a scrape that went wrong.

`openspec/` holds spec-driven change proposals for the larger sessions.

for scraping the new session material use the `scrap-session` skill

## Working style

Commit messages use the `[ERA-V5][muttu]:` prefix — use the `era-commit` skill.
Work happens on `master`. Do not push unless asked.

## Plan Mode

- Make the plan extremely concise. Sacrifice grammar for the sake of concision.
- At the end of each plan, give me a list of unresolved questions to answer, if any.
