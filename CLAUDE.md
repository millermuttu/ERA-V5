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

for scraping the new session material use the `scrap-session` skill

## Working style

Commit messages use the `[ERA-V5][muttu]:` prefix — use the `era-commit` skill.
Work happens on `master`. Do not push unless asked.

## Plan Mode

- Make the plan extremely concise. Sacrifice grammar for the sake of concision.
- At the end of each plan, give me a list of unresolved questions to answer, if any.
