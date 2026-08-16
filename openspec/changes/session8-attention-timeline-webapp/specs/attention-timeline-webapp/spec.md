## Purpose

A hosted, static explanation of how attention got to where it is: every mechanism from the Session 8 lesson placed on a chronological timeline by its real launch date, each one framed as a response to a limitation of what came before, each carrying a sourced date and an honest record of what it buys and what it gives up.

## ADDED Requirements

### Requirement: Baseline-first narrative

The app SHALL open on standard scaled dot-product attention and present it as the mechanism every later entry modifies. No other mechanism SHALL be presented as self-standing. The timeline SHALL begin at the Transformer, so that the baseline is both the first thing read and the earliest entry; RNN-era attention that predates it SHALL appear only as context inside the baseline card, never as its own entry.

#### Scenario: First screen

- **WHEN** a reader loads the app with no fragment
- **THEN** the first mechanism presented is standard scaled dot-product attention, showing the full pipeline Q×K → scores → scale → mask → softmax → weighted sum of V

#### Scenario: Every later mechanism is framed as a response

- **WHEN** a reader opens any mechanism other than the baseline
- **THEN** its card states the problem that existed before it, and that problem refers to a mechanism that appears earlier in the timeline

### Requirement: Chronological ordering by launch date

The app SHALL order mechanisms by the date the technique was published or released, never by the order the lesson taught them and never grouped into a taxonomy first.

#### Scenario: Timeline order

- **WHEN** the timeline is rendered
- **THEN** entries appear in non-decreasing order of their launch date, and each entry displays that date

### Requirement: Sourced and verified chronology

Every mechanism SHALL carry a primary source and an explicit verification state. A date SHALL NOT be presented as established unless it was checked against that primary source.

#### Scenario: Paper-backed mechanism

- **WHEN** a mechanism has a paper as its primary source
- **THEN** the card shows the source title and a link to it, and the displayed date equals the source's first-version publication date

#### Scenario: Mechanism with no primary source

- **WHEN** a mechanism has no public primary source — including DroPE, whose algorithm the lesson states the record does not establish
- **THEN** the card renders an unverified badge naming what the source actually is, and does not present its date or mechanism as established fact

#### Scenario: Sources are listed for submission

- **WHEN** a reader opens the app's README
- **THEN** it lists every mechanism with its source and verification state, matching the app's data exactly

### Requirement: Honest per-mechanism trade-off record

Every mechanism SHALL answer three questions: what it buys, what it gives up, and when it would actually be chosen. A mechanism SHALL NOT be presented with advantages and no cost.

#### Scenario: Complete card

- **WHEN** a reader opens any mechanism card
- **THEN** the card shows at least one thing the mechanism buys, at least one thing it gives up, and a statement of the workload for which it is the right choice

#### Scenario: Missing trade-off is caught

- **WHEN** the app's self-check runs against a mechanism record missing any of the three answers
- **THEN** the self-check fails and names the offending mechanism

### Requirement: Minimum mechanism coverage

The app SHALL cover at least the mechanisms the assignment names: standard attention, learned absolute positions, sinusoidal positions, RoPE, ALiBi, MQA, GQA, sliding window, attention sinks, NTK-aware scaling, YaRN, linear attention, the delta rule and Gated DeltaNet, MLA, sparse and top-k attention, DeepSeek's compressed sparse attention, and DroPE.

The timeline SHALL consist of the entries below, opening at the Transformer. The source column is the contract; the displayed date is whatever the verification of that source establishes, so a date correction changes the ordering rather than breaking this requirement.

| # | mechanism | primary source |
|---|---|---|
| 1 | scaled dot-product attention, multi-head, sinusoidal positions — the baseline | arXiv 1706.03762 |
| 2 | relative position representations | arXiv 1803.02155 |
| 3 | learned absolute position tables | arXiv 1810.04805 |
| 4 | segment recurrence across contexts (Transformer-XL) | arXiv 1901.02860 |
| 5 | Sparse Transformer, strided and fixed patterns | arXiv 1904.10509 |
| 6 | multi-query attention | arXiv 1911.02150 |
| 7 | sliding window with global tokens (Longformer) | arXiv 2004.05150 |
| 8 | linear attention, the kernel regrouping | arXiv 2006.16236 |
| 9 | Performer, FAVOR+ | arXiv 2009.14794 |
| 10 | the delta rule, fast-weight programmers | arXiv 2102.11174 |
| 11 | RoPE | arXiv 2104.09864 |
| 12 | ALiBi | arXiv 2108.12409 |
| 13 | FlashAttention | arXiv 2205.14135 |
| 14 | grouped-query attention | arXiv 2305.13245 |
| 15 | NTK-aware base scaling | community post, no paper |
| 16 | YaRN | arXiv 2309.00071 |
| 17 | attention sinks, StreamingLLM | arXiv 2309.17453 |
| 18 | selective state space (Mamba) | arXiv 2312.00752 |
| 19 | multi-head latent attention | arXiv 2405.04434 |
| 20 | parallelizable DeltaNet | arXiv 2406.06484 |
| 21 | Gated DeltaNet | arXiv 2412.06464 |
| 22 | natively trainable sparse attention | arXiv 2502.11089 |
| 23 | DeepSeek sparse attention | release note, no paper |
| 24 | DroPE | course cookbook only |

Entries beyond this set MAY be added; the assignment counts a relevant mechanism found outside class in the author's favour, provided it carries the same date, source and trade-off record as the rest.

#### Scenario: Coverage check

- **WHEN** the set of mechanism entries is compared against the assignment's named list
- **THEN** every named mechanism is present as its own entry with a date, a source and a complete trade-off record

#### Scenario: The timeline opens at the Transformer

- **WHEN** the entries are sorted by date
- **THEN** the baseline scaled dot-product entry is first, and no entry carries a date earlier than its source's publication

### Requirement: Demos compute real arithmetic

Where a mechanism carries an interactive demo, that demo SHALL compute its numbers in the page from the mechanism's actual arithmetic rather than displaying pre-written results, and SHALL agree with the figures stated in the lesson.

#### Scenario: Causal mask

- **WHEN** a reader turns the causal mask off in the baseline attention demo
- **THEN** attention weight appears on tokens later than the current one, and turning it back on returns those weights to exactly zero

#### Scenario: Regrouping matches only without softmax

- **WHEN** a reader compares the direct route against the pre-built-state route with softmax off
- **THEN** both produce the same output; and **WHEN** softmax is switched on, the two routes no longer agree

#### Scenario: Cache figures match the lesson

- **WHEN** the KV-cache demo is set to 48 layers, 8 KV heads, head dimension 128, bf16 and 32,768 tokens
- **THEN** it reports approximately 6.44 GB for one conversation and approximately 51.54 GB for eight

### Requirement: Runs as static files with no build step

The app SHALL run from static files served over HTTP with no build, no package manager and no third-party runtime dependency, so it can be hosted by copying the folder.

#### Scenario: Local serve

- **WHEN** the app folder is served by any static file server and opened in a browser
- **THEN** the timeline and every demo work with no console errors and no request to a third-party host other than the web font stylesheet

#### Scenario: Hosted under a sub-path

- **WHEN** the app is served from a sub-path rather than a domain root
- **THEN** all of its assets resolve and no asset request 404s

### Requirement: Self-check on demand

The app SHALL expose an on-demand self-check that verifies both its arithmetic and the integrity of its chronology data, and reports pass or fail visibly.

#### Scenario: Running the self-check

- **WHEN** a reader loads the app with the self-check enabled
- **THEN** the app runs its assertions and displays a visible pass or fail result naming any assertion that failed

#### Scenario: Data integrity

- **WHEN** the self-check runs
- **THEN** it verifies that mechanism ids are unique, that every date parses, that every entry has a source and all three trade-off answers, and that every demo reference resolves to a demo that exists
