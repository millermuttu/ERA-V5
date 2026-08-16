## Context

See `proposal.md` — Why, and the specs for what the app must do.

A first implementation was built and rejected: it rendered static tables and prose with six small
demos beside them. The reason it failed is worth recording, because it is the constraint that shapes
everything here. The course's own widget (`Session8/reference/widgets/widget_0b_attention_flow.html`)
runs an actual transformer — 32 dimensions, 4 heads, 2 blocks — over a sentence the reader can edit,
draws the ribbons from a token into its query, key and value, and steps head by head and block by
block. Next to that, a table of six precomputed numbers reads as a description of a mechanism rather
than the mechanism itself. The bar is a live model, and the deliverable has to clear it.

Constraints that follow:

- No JS toolchain in the repo, and none is being added: vanilla ES modules, no build.
- Everything on screen has to come from a forward pass or a cost formula, which means the model must
  be small enough to re-run on every keystroke.
- 26 mechanisms is too many to hand-build as separate widgets and too varied to force through one
  rigid template. The split has to fall somewhere sensible.

## Goals / Non-Goals

**Goals:**

- One model, executing, that every mechanism plugs into — so the comparison between mechanisms is a
  controlled experiment rather than 26 unrelated drawings.
- A mechanism is *implemented*, not depicted: if the card says top-k reads fewer keys, the model read
  fewer keys to produce the numbers on that card.
- The reader's own sentence flows through everything, including the statement of what a mechanism
  cannot reach.
- Every claim about cost is a number that moved while the reader watched.

**Non-Goals:**

- Not a trained model. The weights are seeded noise; the predictions are structurally real and
  semantically meaningless, and the app says so rather than implying competence it does not have.
- Not a benchmark. Nothing here measures quality, and no card claims a mechanism is better because
  the toy behaved a certain way.
- Not a reimplementation of the course widget, and it does not import from `reference/`.

## Decisions

**A real toy transformer, sized to survive a keystroke.** 32 dimensions, 4 heads, 2 blocks, a fixed
small vocabulary, seeded weights — the same shape the reference widget proves is affordable. A
forward pass over ~24 tokens is a few hundred thousand multiplies, fast enough to run synchronously
on input. Alternative considered: a larger model for more convincing predictions. Rejected: the
predictions are noise either way, and responsiveness is a stated requirement.

**The mixer is an interface; every mechanism is an implementation of it.** The block asks a mixer for
new token representations. Two families implement it: `softmaxMixer`, parameterised by a readability
rule, a position scheme, and a key/value sharing scheme; and `stateMixer`, parameterised by a write
rule and a decay. That single seam covers 24 of the 26 entries. It is not a speculative abstraction —
it exists because two dozen concrete callers need it, which is exactly when an interface earns its
place.

**Mechanisms are data plus a small function, held next to their prose.** Each entry supplies its
`mixer`, its default parameters, the controls the card exposes, and a `cost(cfg)`. The renderer knows
nothing about any specific mechanism; adding one is adding a record.

**FlashAttention and the extension methods get honest special handling.** FlashAttention changes no
arithmetic, so it runs the baseline mixer and differs only in the cost model and a memory-traffic
view — the card says exactly that, which is the point of including it. Position interpolation,
NTK-aware scaling, YaRN and DroPE change the position scheme rather than the readability rule, so
their cards lead with the position view and the trained-versus-target boundary. DroPE has no
algorithm in the record, so it renders the boundary and the reported 32× and refuses to simulate a
mechanism nobody published.

**Views compose per card.** Six view modules — attention grid, token flow ribbons, position curve,
state matrix, cost ledger, prediction bars — and a card lists the ones its mechanism needs. Every
view also renders the baseline's version of itself for comparison, since the spec requires the
side-by-side everywhere.

**A deck, not a scroll.** One concept fills the view; a timeline slider along the top carries the
whole chronology, and next/previous controls plus the arrow keys step through it. This replaces the
earlier plan of stacking every card in the document flow. Two reasons it is better here: a card that
runs a live model with its own controls wants the reader's whole attention, and only one concept
computing at a time makes the performance requirement fall out for free rather than needing an
observer to enforce it. The slider is also where the chronology lives — year ticks and entry spacing
show the field's bursts and quiet stretches, which is the drift the assignment asks to be visible.
The URL fragment names the current concept and is kept in step with `history.pushState`, so links,
sharing and the browser's back control all work. Alternative considered: keeping the long scroll and
adding a sticky rail. Rejected — 26 live cards in one document is both slower and a worse reading
experience than the thing the reader actually asked for.

**Two registers of trade-off, both required.** The card carries the technical record the assignment
demands — what it buys, what it gives up, when to choose it — and the foot of the card carries the
same verdict in plain words, no formulas, no unexplained acronyms. They are separate fields on the
record, written separately, and the self-check compares them for count so a plain verdict cannot
quietly drop a cost. This is not duplication: the technical record is for a reader who has done the
course, the plain one is for the friend who asked "how does attention work now?" — which is the
audience the assignment names.

**One shared runner, one shared input.** The sentence, the playback position and the head/layer
selection live in one place; the mounted concept subscribes. The sentence survives moving
between concepts, which is what makes the deck feel like one apparatus rather than 26 toys.

**Only the current concept computes.** The deck makes this automatic: one concept is mounted at a
time, so a keystroke re-runs one forward pass, not 26. No observer, no scheduler, no work-slicing.

**Degenerate-setting equivalence is the correctness test.** GQA with one query head per group, top-k
with k = T, a window as wide as the context, blocks of one token — each must reproduce the baseline
exactly. If a mechanism is faked, that assertion fails. This is what makes the self-check worth more
than a smoke test.

**Dates are not re-verified.** The 22 paper-backed dates were checked against the arXiv API earlier
in this change, which corrected two of them (MQA to 2019-11-06, parallelizable DeltaNet to
2024-06-10). That verification stands and its results are carried into the rebuilt data file; the
three sourceless entries stay unverified with their provenance named.

**One concept at a time, research before code.** The build order is per mechanism, not per layer of
the app: read the primary source, write `docs/research/<id>.md`, build that one card on the live
model, verify it in a browser, stop for review, next. The note records the mechanism precisely, the
parameters that matter, what the reader must be able to do, and what the source does not establish —
so the card is implemented from the paper rather than from recall, and the note becomes the citation
in the README. The cost is that the shared model, mixer seam and cost model must exist before
concept one; that is the foundation phase, and nothing else starts until a forward pass runs.

## Risks / Trade-offs

- **A live model that is slow makes the whole page feel broken** → small model, visible-card-only
  computation, and a typing check in the task list rather than as an afterthought.
- **Seeded-noise predictions could be mistaken for a model that knows something** → the prediction
  view is labelled as untrained throughout, and no card draws a quality conclusion from it.
- **The mixer seam could be bent out of shape by the two or three mechanisms that do not fit** →
  those are handled explicitly (FlashAttention, DroPE) rather than by widening the interface until it
  fits everything and means nothing.
- **Scale honesty: a 24-token toy cannot show a 256K cache** → the cost ledger runs the real serving
  formula with its own inputs, and is checked against the lesson's 6.44 GB / 51.54 GB, while the
  visuals stay at toy scale and say which is which.
- **Rebuilding from an empty folder discards work that was correct** → the prose and the verified
  dates are carried forward from the previous commit rather than rewritten from memory, and the old
  version stays in git history.
- **Twenty-five live configurations is a lot of surface to get wrong quietly** → the degenerate-
  setting assertions cover every mechanism, so a broken one fails loudly.

## Migration Plan

`Session8/webapp/` is deleted and rebuilt. The previous implementation remains in git at `0668e67`
and can be restored with `git checkout 0668e67 -- Session8/webapp` if anything is needed from it.
Order: model, then mixers, then the mechanism registry, then views, then cards, then playback, then
the self-check, then docs. The Pages workflow and the root README row are already in place and do not
change.
