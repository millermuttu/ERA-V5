## Purpose

A hosted explanation of how attention got to where it is, built around a real transformer that runs
in the reader's browser. The timeline is chronological, the arithmetic is live, and every mechanism
on it is swapped into the same running model so that what each one changed is something the reader
watches happen rather than something the page asserts.

## ADDED Requirements

### Requirement: A real model runs in the page

The app SHALL contain a working transformer — embeddings, multi-head attention, feed-forward blocks,
and an output distribution — that executes in the browser on text the reader supplies. Every number,
grid, bar and curve the app displays SHALL be produced by that execution. No displayed quantity may
be a hard-coded result, a recorded trace, or a drawing of what the computation would have looked
like.

#### Scenario: The reader changes the input

- **WHEN** a reader edits the input sentence
- **THEN** the token sequence, the attention scores, the per-head patterns, the cost readouts and the
  next-token distribution all recompute from the new text

#### Scenario: Nothing is canned

- **WHEN** any displayed value is traced back to its origin
- **THEN** it is the output of the model's forward pass or of a cost formula evaluated on the current
  configuration, never a literal transcribed from the lesson or from a paper

#### Scenario: The model is deterministic

- **WHEN** the page is reloaded with the same input
- **THEN** every number is identical, because the weights come from a seeded generator rather than an
  unseeded one

### Requirement: One spine, every mechanism a configuration of it

Every mechanism on the timeline SHALL be implemented as a configuration of the same running model —
a way of deciding which keys a query may read, how position enters the score, how many key/value
heads are stored, or how the past is compressed into state — and not as a separate illustration
alongside the model.

#### Scenario: Swapping a mechanism

- **WHEN** a reader moves from one card to another
- **THEN** the same model, on the same input, is re-run under the new mechanism, and the attention
  pattern, the costs and the prediction change accordingly

#### Scenario: A mechanism reduces to the baseline at its degenerate setting

- **WHEN** a mechanism is set to the parameters that make it equivalent to full attention — grouped
  query attention with one query head per group, top-k selection with k equal to the context length,
  a sliding window as wide as the context, block compression with one token per block
- **THEN** its output matches the baseline's output to within floating-point tolerance

#### Scenario: Mechanisms that change no arithmetic say so

- **WHEN** a mechanism alters only how the computation meets the hardware rather than what it
  computes, as FlashAttention does
- **THEN** the app shows its output as identical to the baseline and locates the change in the cost
  ledger and the memory-traffic view instead

### Requirement: Every mechanism is shown against the baseline

Every non-baseline card SHALL present its mechanism next to plain scaled dot-product attention on the
same input, so that what changed is visible as a difference rather than described in prose.

#### Scenario: The comparison is on screen

- **WHEN** a reader opens any mechanism card
- **THEN** the baseline's behaviour on the current input is displayed alongside the mechanism's, and
  the positions, weights or costs that differ are marked

#### Scenario: What was lost is named

- **WHEN** a mechanism cannot reach a position that the baseline attends to
- **THEN** the app identifies that position in the reader's own sentence and states that it is
  unreachable, rather than only reporting an aggregate

### Requirement: Playback

The app SHALL let the reader advance the computation in steps rather than only seeing its final
state, with at minimum the ability to move through generated tokens one at a time and through the
model's heads and layers.

#### Scenario: Stepping

- **WHEN** a reader steps forward
- **THEN** the display advances by one token, head or layer and the visuals redraw for that step

#### Scenario: Playing and stopping

- **WHEN** a reader starts playback
- **THEN** the steps advance automatically until the reader pauses or the sequence ends, and pausing
  leaves the display on the step reached

#### Scenario: Generation is real

- **WHEN** the reader plays token generation
- **THEN** each new token is sampled from the model's own output distribution and appended to the
  context, and the growing key/value cache is reflected in the cost readout

### Requirement: A live cost ledger

Every card SHALL carry a cost readout — at minimum the key/value cache in bytes, the number of keys
each query reads, and the mixing work per token — recomputed as the reader moves that mechanism's
controls, alongside the same three numbers for the baseline.

#### Scenario: Costs react

- **WHEN** a reader changes a mechanism's parameters
- **THEN** the cost readout updates in the same interaction, without a reload

#### Scenario: Serving scale is reachable from the toy

- **WHEN** the reader raises the cost ledger to a serving configuration
- **THEN** the same formulas produce the figures the lesson states — approximately 6.44 GB for one
  conversation and 51.54 GB for eight at 48 layers, 8 key/value heads, head dimension 128, bf16 and
  32,768 tokens

### Requirement: Baseline-first narrative

The app SHALL open on standard scaled dot-product attention and present it as the mechanism every
later entry modifies. No other mechanism SHALL be presented as self-standing. The timeline SHALL
begin at the Transformer, so that the baseline is both the first thing read and the earliest entry;
attention that predates it SHALL appear only as context inside the baseline card.

#### Scenario: First screen

- **WHEN** a reader loads the app with no fragment
- **THEN** the first mechanism presented is standard scaled dot-product attention, showing the full
  pipeline Q×K → scores → scale → mask → softmax → weighted sum of V running on the current input

#### Scenario: The six stages are individually visible

- **WHEN** a reader works through the baseline card
- **THEN** each of Q×K, the scores, the scaling, the mask, the softmax and the weighted sum of V can
  be inspected as its own stage with its own numbers, rather than only the finished attention pattern

#### Scenario: Every later mechanism is framed as a response

- **WHEN** a reader opens any mechanism other than the baseline
- **THEN** its card states the problem that existed before it, and that problem refers to a mechanism
  that appears earlier in the timeline

### Requirement: Chronological ordering by launch date

The app SHALL order mechanisms by the date the technique was published or released, never by the
order the lesson taught them. Time SHALL be the only axis the timeline is organised on: the app SHALL
NOT group entries into categories, families or sections ahead of the chronology, and any label
describing what kind of pressure a mechanism answers SHALL be secondary to its position in time.

#### Scenario: Timeline order

- **WHEN** the timeline is rendered
- **THEN** entries appear in non-decreasing order of their launch date, and each entry displays that
  date

#### Scenario: No taxonomy first

- **WHEN** the reader moves through the concepts from first to last
- **THEN** entries of the same kind are not collected together, and the reader encounters them
  interleaved exactly as the dates dictate

### Requirement: One concept at a time

The app SHALL present a single mechanism at a time rather than a continuous scroll of all of them, so
that a reader can settle on one concept, work its controls, and move on deliberately.

#### Scenario: Only the current concept is shown

- **WHEN** the app is open
- **THEN** exactly one mechanism occupies the view, with its own visuals, controls and verdict, and
  the others are not competing for attention below it

#### Scenario: Moving between concepts

- **WHEN** a reader activates the next or previous control
- **THEN** the app advances to the adjacent mechanism in date order, and the reader can tell from the
  display where they are in the sequence and how many remain

#### Scenario: Sliding along the timeline

- **WHEN** a reader drags the timeline slider
- **THEN** the displayed concept follows it, and the slider's own layout shows the chronology — where
  the years fall and how the entries are spread across them

#### Scenario: A concept can be linked to

- **WHEN** a reader opens a URL carrying a mechanism's identifier, or shares the URL while on a
  concept
- **THEN** the app opens on that concept, and moving between concepts keeps the URL in step so the
  browser's back control returns to the previous one

#### Scenario: The keyboard works

- **WHEN** a reader presses the left or right arrow key outside a text field
- **THEN** the app moves to the previous or next concept

### Requirement: A plain-language verdict on every concept

The foot of every concept SHALL carry its pros and cons written in everyday language — no formulas,
no symbols, and no acronym or piece of jargon that the sentence does not itself explain — so that a
reader who has not taken the course can still say what the mechanism won and what it lost.

#### Scenario: The verdict is readable without the theory

- **WHEN** a reader who has not read the lesson reaches the foot of a concept
- **THEN** they find its advantages and its costs stated in ordinary words, and can restate what the
  mechanism gives and takes without using the technical vocabulary above it

#### Scenario: Plain does not mean vague

- **WHEN** the plain-language verdict is compared with the technical trade-off record on the same
  card
- **THEN** they agree: the verdict states the same advantages and the same costs in simpler words,
  and does not omit a cost or add a benefit the technical record does not support

#### Scenario: Every concept has one

- **WHEN** the self-check runs
- **THEN** it fails and names any mechanism whose plain-language verdict is missing, or whose verdict
  offers advantages with no costs

### Requirement: The progression is legible

The timeline SHALL make the field's movement visible, not merely list mechanisms in date order. Each
entry SHALL connect backwards to the limitation it answered and forwards to what its own cost forced
somebody to fix next, and the app SHALL make the drift in the field's priorities over time readable —
exact global attention, then cheaper decoding memory, then better position handling, then longer
contexts, then recurrent state returning, then sparsity returning, then compression growing more
aggressive.

#### Scenario: Backwards and forwards

- **WHEN** a reader opens any entry other than the last
- **THEN** the card names the earlier mechanism whose limitation it answers, and names what its own
  trade-off left for later work to fix, with both references resolving to entries on the timeline

#### Scenario: The drift is visible

- **WHEN** a reader looks at the timeline slider as a whole
- **THEN** the changing priority of the field over time can be read off it, without the entries having
  been grouped by priority

### Requirement: Interaction has to teach

Every control the app exposes SHALL change a quantity that is part of the mechanism being explained.
Interaction that does not change what the mechanism does or costs SHALL NOT be added.

#### Scenario: Every control earns its place

- **WHEN** a reader moves any control on any card
- **THEN** a displayed quantity belonging to that mechanism's behaviour or its trade-off changes in
  response

#### Scenario: Clarity outranks cleverness

- **WHEN** an interaction would be more impressive than instructive
- **THEN** the app presents the mechanism plainly instead, since a clear static explanation is worth
  more than an interaction that does not teach the mechanism

### Requirement: Sourced and verified chronology

Every mechanism SHALL carry a primary source and an explicit verification state. A date SHALL NOT be
presented as established unless it was checked against that primary source.

#### Scenario: Paper-backed mechanism

- **WHEN** a mechanism has a paper as its primary source
- **THEN** the card shows the source title and a link to it, and the displayed date equals the
  source's first-version publication date

#### Scenario: Mechanism with no primary source

- **WHEN** a mechanism has no public primary source — including DroPE, whose algorithm the lesson
  states the record does not establish
- **THEN** the card renders an unverified badge naming what the source actually is, and does not
  present its date or mechanism as established fact

#### Scenario: Sources are listed for submission

- **WHEN** a reader opens the app's README
- **THEN** it lists every mechanism with its source and verification state, matching the app's data
  exactly

### Requirement: Lesson claims are checked, not inherited

Where the lesson states something that can be checked against a primary source, the app SHALL check
it and report what it found. A claim that cannot be confirmed SHALL be attributed to the course
record rather than presented as public fact, and a claim that disagrees with the primary source SHALL
be corrected on the card with both versions shown.

#### Scenario: A checkable claim is confirmed

- **WHEN** the lesson states a figure that a primary source or a formula can confirm, such as the KV
  cache at a stated configuration
- **THEN** the app reproduces it from its own computation and says so

#### Scenario: A claim cannot be confirmed

- **WHEN** the lesson attributes a mechanism or a number to a model whose public record does not
  confirm it
- **THEN** the card names what the public sources do establish, names what the lesson says, and does
  not merge the two into a single unattributed statement

### Requirement: Submission artefacts

The repository SHALL carry a README that a reader can use to reach and check the work: what the app
is, the live link, the repository link, how to run it locally, and the full chronology with one
source per entry.

#### Scenario: The README stands alone

- **WHEN** somebody reads the README without opening the app
- **THEN** they can find the live link, the repository, the run instructions, and every mechanism's
  date, source and verification state

### Requirement: Honest per-mechanism trade-off record

Every mechanism SHALL answer three questions: what it buys, what it gives up, and when it would
actually be chosen. A mechanism SHALL NOT be presented with advantages and no cost.

#### Scenario: Complete card

- **WHEN** a reader opens any mechanism card
- **THEN** the card shows at least one thing the mechanism buys, at least one thing it gives up, and
  a statement of the workload for which it is the right choice

#### Scenario: Missing trade-off is caught

- **WHEN** the self-check runs against a mechanism record missing any of the three answers
- **THEN** the self-check fails and names the offending mechanism

### Requirement: Minimum mechanism coverage

The app SHALL cover at least the mechanisms the assignment names: standard attention, learned
absolute positions, sinusoidal positions, RoPE, ALiBi, MQA, GQA, sliding window, attention sinks,
NTK-aware scaling, YaRN, linear attention, the delta rule and Gated DeltaNet, MLA, sparse and top-k
attention, DeepSeek's compressed sparse attention, and DroPE.

The timeline SHALL consist of the entries below, opening at the Transformer. The source column is
the contract; the displayed date is whatever the verification of that source establishes.

| # | mechanism | primary source |
|---|---|---|
| 1 | scaled dot-product attention, multi-head — the baseline | arXiv 1706.03762 |
| 2 | sinusoidal position encoding | arXiv 1706.03762 |
| 3 | relative position representations | arXiv 1803.02155 |
| 4 | learned absolute position tables | arXiv 1810.04805 |
| 5 | segment recurrence across contexts (Transformer-XL) | arXiv 1901.02860 |
| 6 | Sparse Transformer, strided and fixed patterns | arXiv 1904.10509 |
| 7 | multi-query attention | arXiv 1911.02150 |
| 8 | sliding window with global tokens (Longformer) | arXiv 2004.05150 |
| 9 | linear attention, the kernel regrouping | arXiv 2006.16236 |
| 10 | Performer, FAVOR+ | arXiv 2009.14794 |
| 11 | the delta rule, fast-weight programmers | arXiv 2102.11174 |
| 12 | RoPE | arXiv 2104.09864 |
| 13 | ALiBi | arXiv 2108.12409 |
| 14 | FlashAttention | arXiv 2205.14135 |
| 15 | grouped-query attention | arXiv 2305.13245 |
| 16 | position interpolation | arXiv 2306.15595 |
| 17 | NTK-aware base scaling | community post, no paper |
| 18 | YaRN | arXiv 2309.00071 |
| 19 | attention sinks, StreamingLLM | arXiv 2309.17453 |
| 20 | selective state space (Mamba) | arXiv 2312.00752 |
| 21 | multi-head latent attention | arXiv 2405.04434 |
| 22 | parallelizable DeltaNet | arXiv 2406.06484 |
| 23 | Gated DeltaNet | arXiv 2412.06464 |
| 24 | natively trainable sparse attention | arXiv 2502.11089 |
| 25 | DeepSeek sparse attention | release note, no paper |
| 26 | DroPE | course cookbook only |

Entries beyond this set MAY be added, provided they carry the same date, source and trade-off record.

#### Scenario: Coverage check

- **WHEN** the set of mechanism entries is compared against the assignment's named list
- **THEN** every named mechanism is present as its own entry with a date, a source, a complete
  trade-off record and a live configuration of the model

#### Scenario: The timeline opens at the Transformer

- **WHEN** the entries are sorted by date
- **THEN** the baseline scaled dot-product entry is first, and no entry carries a date earlier than
  its source's publication

### Requirement: Responsive on an ordinary machine

The model SHALL be small enough that editing the input or moving a control redraws without a visible
stall, and the app SHALL remain usable on a narrow screen.

#### Scenario: Typing stays smooth

- **WHEN** a reader types into the input on a mid-range laptop
- **THEN** the visuals keep up with the typing, with no frozen frame long enough to read as a hang

#### Scenario: Narrow screens

- **WHEN** the page is viewed at a phone width
- **THEN** the content reflows, wide visuals scroll inside their own container, and the page body
  itself does not scroll sideways

### Requirement: Runs as static files with no build step

The app SHALL run from static files served over HTTP with no build, no package manager and no
third-party runtime dependency, so it can be hosted by copying the folder.

#### Scenario: Local serve

- **WHEN** the app folder is served by any static file server and opened in a browser
- **THEN** the timeline and every mechanism work with no console errors and no request to a
  third-party host other than the web font stylesheet

#### Scenario: Hosted under a sub-path

- **WHEN** the app is served from a sub-path rather than a domain root
- **THEN** all of its assets resolve and no asset request 404s

### Requirement: Self-check on demand

The app SHALL expose an on-demand self-check that verifies the model, the mechanism implementations
and the integrity of the chronology, and reports pass or fail visibly.

#### Scenario: Running the self-check

- **WHEN** a reader loads the app with the self-check enabled
- **THEN** the app runs its assertions and displays a visible pass or fail result naming any
  assertion that failed

#### Scenario: The model is checked, not just the data

- **WHEN** the self-check runs
- **THEN** it verifies that softmax rows sum to one, that masked future weight is exactly zero, that
  each mechanism reduces to the baseline at its degenerate setting, that the softmax-free regrouping
  reproduces the direct computation, that the delta rule corrects where an add-only write
  accumulates, and that the cost formulas reproduce the lesson's serving figures

#### Scenario: Data integrity

- **WHEN** the self-check runs
- **THEN** it verifies that mechanism ids are unique, that every date parses, that every entry has a
  source and all three trade-off answers, and that every entry resolves to a mechanism the model can
  actually run
