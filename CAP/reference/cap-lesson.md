# CAP: Capstone Lab — Plan Sheet

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmcapsession0erav5lab0001/lesson
Captured: 2026-10-26 (sheet says last updated 26 Oct 2026; TSAI edits it during the lab, so re-scrape for changes)

Widgets (1) saved under `widgets/`: `plan_sheet.html`, the whole plan as one interactive page.
Videos embedded on the page: https://www.youtube.com/embed/AGs8EkpOqcw (Studio),
https://www.youtube.com/embed/C2FRJeq8UX8 (GMeet) — the same two as Session 16.

Everything below "Plan sheet" is read out of the `DATA` object in the widget script, not typed by hand.
Student names and the signed pack download links are left out of this file; they are in the widget.

## Page text

This page is the working plan of the ERA V5 training lab. It shows the timeline, the members of each
team, the work of each week, and your sample pack for the data diagnosis task.

- Lab repository: https://github.com/theschoolofai/era-v5-lab
- Team chat: your team group under CAP in the course chat (G1 to G12).
- Training run starts: Saturday 28 November.

*Widget: Lab plan sheet (`plan_sheet`). Timeline, one tab per team, a name search for your teams and your
pack, the sample-pack table and the waitlist.*

## Plan sheet

> Four weeks of team work before our training run starts on Saturday 28 November. Each team has a tab.
> Each week shows what the students do, what the TSAI Lab does, and the gate at the end of the week.

### Rules for every student

> - **The 3-day rule.** A task is due in 3 days. 90% must be done, or you show proof that you are on
>   target. If not, you leave the project, and a waitlist student takes your place. Data tasks get 7 days.
> - **Data first.** Every student starts with the data diagnosis task in week 1, in every team. Your sample
>   pack is in the table below.
> - **Three teams, staggered.** Your core team is your main load for all four weeks. In week 2 you also do
>   a small rotation task in a second team, and in week 3 in a third team. In week 4 everyone works in the
>   core team for dry run 2 and the launch. The goal is that you see as much of a full training run as
>   possible. Some students do only the diagnosis task.
> - **Team chat.** Each team has a group under CAP in the course chat: G1 to G12.
> - **Proof on GitHub.** Publish your results on GitHub. TSAI judges the proof.
> - **No keys.** You do not get cloud keys. Ask for data with the request page. The lab downloads it and
>   gives you sample packs.
> - Gates are on Saturdays: 31 Oct, 14 Nov, 21 Nov and 28 Nov. There is no work during the break, 5 to 11
>   Nov.

### Weeks and key dates

| Week | Dates | Gate |
|---|---|---|
| 1 | Sat 24 Oct to Sat 31 Oct | Gate 1 · Sat 31 Oct |
| 2 | Sun 1 Nov to Sat 14 Nov · break 5 to 11 Nov | Gate 2 · Sat 14 Nov |
| 3 | Sun 15 Nov to Sat 21 Nov | Gate 3 · Sat 21 Nov |
| 4 | Sun 22 Nov to Sat 28 Nov | Gate 4 · Sat 28 Nov · training starts |

| Date | Event |
|---|---|
| 24 Oct | Team work starts |
| 27 Oct | Mid-week check |
| 31 Oct | Gate 1 |
| 5–11 Nov | Break |
| 14 Nov | Gate 2: tokenizer and config frozen |
| 21 Nov | Gate 3: dry run 1 ends |
| 24–26 Nov | Dry run 2 |
| 28 Nov | Gate 4: training starts |

### Week 1 task (the assignment for every student)

> Find your name. Download your pack. Follow `diagnosis/DISEASES.md` in the lab repository:
> github.com/theschoolofai/era-v5-lab. **First 150 labelled rows are due Tue 27 Oct, 23:59 IST. All rows
> and the medicine sheet are due Sat 31 Oct, 23:59 IST.**

### Waitlist

> HOW TO GET OUT OF THE WAITLIST: submit the last 5 assignments (Session 10, 11, 12, 13 and 14) by
> **Wed 4 Nov, 23:59 IST**, before Diwali (Mon 9 Nov). Students who do this get a place in the later
> phases of the capstone.
>
> A waitlist student takes a place when a team member leaves under the 3-day rule. TSAI tells you in the
> chat when you get a place. Until then, you can follow the CAP channel.

Tags in the team rows: **HAVE** we already have it · **BUILD** we build it · **CONFIRM** we must check it ·
**DECIDE** TSAI decides. Only the TSAI Lab changes this sheet.

### Teams

| # | Team | Scope | Core size | Week 2 rotation in | Week 3 rotation in |
|---|---|---|---|---|---|
| 1 | Data A | English, code, math and science data | 12 | 7 | 6 |
| 2 | Data B | Indian languages and code-adjacent text (issues, pull requests, commits, notebooks) | 11 | 7 | 6 |
| 3 | Tokenizer | A new tokenizer for code, English and Indian languages | 5 | 7 | 5 |
| 4 | KE2 | Kronecker Embedding 2 at the input and at the output | 6 | 0 | 0 |
| 5 | Architecture | Model configuration, small-model harness and design experiments | 11 | 6 | 9 |
| 6 | Performance and Kernels | Speed and correctness of each part of the training step | 9 | 8 | 9 |
| 7 | Infrastructure | Job queue, GPU health, checkpoints, parallel layout and dry runs | 6 | 9 | 8 |
| 8 | Monitoring and Records | Metrics, alarms, dashboards, per-token records and on-call | 5 | 9 | 8 |
| 9 | Evaluation | Benchmarks, evaluation harness, benchmark fingerprints and checkpoint evaluation | 8 | 9 | 8 |
| 10 | Post-training and Agentic | SFT and RL strategy, special tokens, chat format, SFT and RL data, SWE environments | 12 | 9 | 10 |
| 11 | Release | Model card, report, weekly public updates, public dashboard text and launch announcement | 2 | 7 | 6 |
| 12 | Quantization and Serving | Quantization tools, score drop at each bit width, automatic quantize flow, public demo and ollama converter | 5 | 9 | 12 |

#### 1. Data A

*English, code, math and science data.* Find the diseases in English, code, math and science data, and write safe cleaning rules.

Timeline bars:

- 2026-10-24 to 2026-10-31: Diagnosis of sample packs
- 2026-10-24 to 2026-11-04: Gold sets and medicine harm tests
- 2026-11-01 to 2026-11-14: Tokenizer sample check and dedup design (break 5 to 11 Nov)
- 2026-11-15 to 2026-11-21: Audit of cleaning at scale
- 2026-11-22 to 2026-11-28: Shard checks for dry run 2 and launch

Milestones:

- 2026-10-27: First 150 labelled rows
- 2026-10-31: Spec v1 frozen
- 2026-11-21: First training shards pass the audit
- 2026-11-28: Data go or no-go

**Week 1 — Diagnosis and spec v1**

- Students:
  - Label all ~500 sample documents with disease codes A to H and evidence.
  - Give 150 labelled rows on 27 Oct and all rows on 31 Oct.
  - Write a medicine sheet with a rule for each disease above five cases.
  - Build the viewer, the gold sets and the harm-test script for each rule.
- TSAI Lab:
  - **HAVE** Sample packs and download links are on this page, from 24 Oct.
  - **HAVE** The disease list, the templates and the viewer are in the lab repository.
- Depends on:
  - Team 9: benchmark fingerprints, 31 Oct
  - Team 2: Indic rules for spec v1
- Gate: Spec v1 is frozen on 31 Oct, and each rule passes its harm test.

**Week 2 — Spec v1 on the tokenizer sample, and dedup design**

- Students:
  - Apply spec v1 to the tokenizer sample on the CPU cluster.
  - Read 200 sample documents per domain. Report diseases that spec v1 missed.
  - Write the dedup design, the source priority list and overlap_matrix.csv.
  - Write shard_format_v1.md. Break 5 to 11 Nov, with no work.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives CPU cluster time for the tokenizer sample.
  - **DECIDE** TSAI approves the dedup design and the source priority list.
- Rotation task (students from other teams):
  - Audit 100 documents of one source after the medicine run.
  - Report each new disease with evidence to the Data A lead.
- Depends on:
  - Team 9: benchmark fingerprints, 31 Oct
  - Team 3: tokenizer freeze, 14 Nov
- Gate: The reading check finds no new disease above 1%, and TSAI approves the dedup design.

**Week 3 — Audit of cleaning at scale**

- Students:
  - Run cleaning, dedup, decontamination and tokenizing at scale, 15 to 21 Nov.
  - Make daily audit packs. Read 100 cleaned documents per source each day.
  - Compare text before and after cleaning. The rule must change only the disease.
  - Decode 20 documents per shard and check each shard manifest.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives CPU cluster time for cleaning at scale.
  - **DECIDE** TSAI sets the data mix of the first training weeks.
- Rotation task (students from other teams):
  - Check 2 tokenized shards: manifest fields, hashes and token counts.
  - File each problem as an issue in the lab repository.
- Depends on:
  - Team 3: frozen tokenizer with hash, 14 Nov
  - Team 7: data position format in checkpoint
- Gate: The first training shards pass the audit and match their manifests.

**Week 4 — Shards for dry run 2 and the training run**

- Students:
  - Make the shards for the training weeks after launch, before each week starts.
  - Check dry run 2 shards on 24 Nov: counts, hashes and manifests.
  - Check that the loader resumes at the correct data position after the failure.
  - Write data_card_v1.md and give the data report to TSAI on 28 Nov.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives CPU cluster time for the training-week shards.
  - **DECIDE** TSAI gives go or no-go for data, 28 Nov.
- Depends on:
  - Team 7: dry run 2, 24 Nov
  - Team 6: loader timing
- Gate: Audited shards for the first training weeks are ready, and data is go on 28 Nov.

#### 2. Data B

*Indian languages and code-adjacent text (issues, pull requests, commits, notebooks).* Protect Indian-language text with native readers, and find the diseases in Indic and code-adjacent data.

Timeline bars:

- 2026-10-24 to 2026-10-31: Diagnosis of Indic and code-adjacent packs
- 2026-10-24 to 2026-11-04: Reader list and Indic gold sets
- 2026-11-01 to 2026-11-14: NVIDIA Indic translation check (break 5 to 11 Nov)
- 2026-11-15 to 2026-11-21: Audit of Indic cleaning at scale
- 2026-11-22 to 2026-11-28: Indic shards for dry run 2 and launch

Milestones:

- 2026-10-27: First 150 labelled rows
- 2026-10-31: Spec v1 frozen with Indic rules
- 2026-11-14: Translation quality table per language
- 2026-11-21: Indic shards pass the reader audit

**Week 1 — Diagnosis by native readers**

- Students:
  - Label all of your sample pack with diseases A to H, evidence and hex.
  - Keep ZWJ and ZWNJ always. Check G1 to G4 damage first.
  - Give 150 rows on 27 Oct. Give all rows and medicines on 31 Oct.
  - Build the viewer, the harm-test script and a 200-document gold set per language.
- TSAI Lab:
  - **HAVE** Indic, Malayalam, Odia and code-adjacent sample packs are on this page.
  - **HAVE** Native readers are assigned to every Indic pack.
- Depends on:
  - Team 9: fingerprints with FLORES-200 and IN22
  - Team 1: shared spec v1 format, 31 Oct
- Gate: Each Indic language with readers has full labels, a gold set and medicines that pass the harm test.

**Week 2 — Translation quality and Indic filters**

- Students:
  - Make review packs and rate NVIDIA Indic translations from 1 to 5.
  - Two readers rate the same 20 documents per language. Record the agreement.
  - Measure the romanised Indic share and propose the cap in romanised_policy_v1.md.
  - Build the script-aware filters. Break 5 to 11 Nov, with no work.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gets the NVIDIA Indic set release date.
  - **DECIDE** TSAI sets the native Indic share and the romanised cap.
- Rotation task (students from other teams):
  - If you read an Indian language, label 100 Indic documents.
  - If not, audit 100 code-adjacent documents (issues, commits, notebooks).
- Depends on:
  - NVIDIA: Indic translation set release
  - Team 3: tokenizer freeze, 14 Nov
- Gate: Each language has a translation quality score or a written reason, and the romanised cap is set.

**Week 3 — Audit of Indic and code-adjacent cleaning at scale**

- Students:
  - Run Indic and code-adjacent cleaning at scale on the CPU cluster.
  - Make daily audit packs. Read 50 cleaned documents per language each day.
  - Count ZWJ and ZWNJ before and after cleaning. The counts must match.
  - Check language labels for script-sharing pairs and filters that drop over 20%.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives CPU cluster time for Indic cleaning at scale.
  - **DECIDE** TSAI picks the translated languages that enter the mix.
- Rotation task (students from other teams):
  - Check Indic shard samples for script damage and lost ZWJ or ZWNJ.
  - Give each finding with the document ID and a hex dump.
- Depends on:
  - Team 3: frozen tokenizer with hash, 14 Nov
  - Team 9: fingerprints for decontamination
- Gate: Indic shards keep ZWJ and ZWNJ, pass the reader audit and stay inside approved drop rates.

**Week 4 — Indic shards for dry run 2 and launch**

- Students:
  - Make the Indic and code-adjacent shards for the training weeks after launch.
  - Check dry run 2 shards on 24 Nov. Hashes and manifests must match.
  - Decode 20 documents per language and compare them byte for byte.
  - Write the Indic section of data_card_v1.md. Give the reader list to Team 9.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives CPU cluster time for the Indic shards.
  - **DECIDE** TSAI gives go or no-go for Indic data, 28 Nov.
- Depends on:
  - Team 7: dry run 2, 24 Nov
  - Team 9: Indic benchmark list
- Gate: Audited Indic shards for the first training weeks are ready, and Indic data is go on 28 Nov.

#### 3. Tokenizer

*A new tokenizer for code, English and Indian languages.* Build a tokenizer that beats BrahmicTokenizer-131K on Indic, with no loss on code and English.

Timeline bars:

- 2026-10-24 to 2026-10-31: Baseline reproduction and test harness
- 2026-11-01 to 2026-11-14: Candidate training and freeze (break 5 to 11 Nov)
- 2026-11-15 to 2026-11-21: Tokenization at scale with the frozen tokenizer
- 2026-11-22 to 2026-11-28: Dry run 2 checks and release files

Milestones:

- 2026-10-28: Provisional tokenizer v0 with hash
- 2026-10-31: Special-token list final
- 2026-11-14: Tokenizer frozen

**Week 1 — Baseline, harness and training script**

- Students:
  - Run the verify scripts and reproduce BrahmicTokenizer-131K fertility within 0.001.
  - Add comparator rows for Gemma-4, Qwen, DeepSeek V4, Sarvam and o200k.
  - Build run_harness.py and train_bpe.py with fixed slots per Brahmic script.
  - Make tokenizer v0 with reserved special slots and a sha256 hash, 28 Oct.
- TSAI Lab:
  - **HAVE** BrahmicTokenizer-131K is tokenizer v0. Week 1 runs on your own machines.
  - **HAVE** Pass bar: beat BrahmicTokenizer on mean Indic fertility, at most 2% worse on code and English.
- Depends on:
  - Team 10: special-token list, 31 Oct
  - TSAI Lab: sample packs, 24 Oct
- Gate: The baseline matches the reference, and train_bpe.py output passes the verify scripts.

**Week 2 — Train candidates and freeze on 14 Nov**

- Students:
  - Train candidates C1 to C3 on the tokenizer sample, 1 to 4 Nov.
  - Run the harness on each candidate. Add one row to COMPARE.md.
  - Send each candidate to Team 4 for collisions. Break 5 to 11 Nov.
  - Write freeze_report.md and the release folder with SHA256SUMS, 14 Nov.
- TSAI Lab:
  - **DECIDE** TSAI freezes the tokenizer on 14 Nov from COMPARE.md.
  - **DECIDE** If no candidate passes, C0 with our special slots is frozen.
- Rotation task (students from other teams):
  - Run the fertility harness on one candidate for one language.
  - Post the result table in the G3 group.
- Depends on:
  - Team 1: tokenizer sample, 1 Nov
  - Team 4: collision count for each candidate
- Gate: The frozen tokenizer beats BrahmicTokenizer-131K on Indic, has no loss on code and English, and has a hash.

**Week 3 — Tokenization at scale**

- Students:
  - Tokenize the cleaned data at scale with the frozen tokenizer, 15 to 21 Nov.
  - Measure tokenizer speed on Graviton c7g in MB per second.
  - Check round trip on 1,000 documents per domain and the hash per shard.
  - Test the chat format with Team 10 and write the tokenizer card.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives CPU cluster time for tokenizing at scale.
  - **DECIDE** Token IDs stay fixed after 14 Nov. TSAI approves each fix.
- Rotation task (students from other teams):
  - Measure tokenizer speed on 1 GB of text.
  - Compare the speed with BrahmicTokenizer-131K.
- Depends on:
  - Team 1 and Team 2: cleaned data
  - Team 10: chat format with loss masks
- Gate: The first token shards round-trip with 0 failures and record the frozen hash.

**Week 4 — Dry run 2 and release files**

- Students:
  - Check that the trainer, eval harness and chat format use one tokenizer hash.
  - Run the special-token test on SFT samples from Team 10.
  - Make the Hugging Face release files and the comparison table with Team 11.
  - Put the tokenizer card and harness reports on GitHub.
- TSAI Lab:
  - **DECIDE** TSAI approves the public release date of the tokenizer.
- Depends on:
  - Team 7: dry run 2, 24 Nov
  - Team 11: report outline
- Gate: All teams use the frozen tokenizer hash in dry run 2, and the release files are ready.

#### 4. KE2

*Kronecker Embedding 2 at the input and at the output.* Prove that the Kronecker code encodes and decodes every token exactly, then test the KE2 output head.

Timeline bars:

- 2026-10-24 to 2026-10-31: Forward encode proof and collision count
- 2026-10-28 to 2026-11-04: Deterministic backward decode and margins
- 2026-11-01 to 2026-11-14: Revalidate the TSAI KE2 head and find bugs (break 5 to 11 Nov)
- 2026-11-15 to 2026-11-21: Chosen path in the trainer and dry run 1
- 2026-11-22 to 2026-11-28: Dry run 2 checks

Milestones:

- 2026-10-31: Forward encode proof
- 2026-11-14: KE2 decision

**Week 1 — Forward encode proof and collisions**

- Students:
  - Build decode and raw byte tables for P=32 and P=64.
  - Count collisions per script class, with a cause for each group.
  - Write Lemma 1, Lemma 2 and Theorem 1 in proof.md, with numerical checks.
  - Decode every token ID on CPU three times. The three hashes must match.
- TSAI Lab:
  - **HAVE** Week-1 proofs run on CPU on your own machines.
  - **HAVE** The kronecker-embeddings repository is the starting code.
- Depends on:
  - Team 3: provisional tokenizer, 28 Oct
  - Team 10: special-token list, 31 Oct
- Gate: The forward encode proof is checked, and each token has a unique code or a known collision cause.

**Week 2 — Backward decode and revalidation**

- Students:
  - Decode every token on GPU three times with batch sizes 1, 7 and 4,096.
  - Add Theorem 2, Theorem 3 and the margin table to proof.md.
  - Re-run the TSAI KE2 head results from the code shared on 28 Oct, with Team 5. Report each difference and each bug.
  - Count collisions for each Team 3 candidate. Break 5 to 11 Nov.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives GPU time for the small-model runs.
  - **DECIDE** TSAI makes the KE2 decision on 14 Nov.
- Depends on:
  - Team 3: candidates and frozen tokenizer
  - Team 5: small-model harness
- Gate: Decode is deterministic on CPU and GPU for every token, and TSAI makes the KE2 decision.

**Week 3 — The chosen path in the trainer**

- Students:
  - Put the chosen input and output path into the trainer with Team 5.
  - Write correctness tests with Team 6 for every token ID.
  - Build the code table for the frozen tokenizer in bf16, with 0 special-token collisions.
  - Time encode and decode per step in the Team 6 timing harness.
- TSAI Lab:
  - **DECIDE** TSAI sets the tau value for a valid decode.
- Depends on:
  - Team 6: timing harness and correctness tests
  - Team 7: dry run 1
- Gate: Dry run 1 runs with the chosen path, and the correctness tests pass on every token ID.

**Week 4 — Dry run 2**

- Students:
  - Watch the decode metrics in dry run 2. Report each change in rates.
  - After the planned failure, check that the code table keeps its sha256.
  - Make the output step faster. Each change needs a passing correctness test.
  - Write the KE2 report section and a public proof summary with Team 11.
- TSAI Lab:
  - **DECIDE** TSAI approves the public proof summary before release.
  - **DECIDE** TSAI gives go or no-go for the chosen path, 28 Nov.
- Depends on:
  - Team 7: dry run 2, 24 Nov
  - Team 8: metric alarms
- Gate: Dry run 2 passes with the chosen path after the planned failure, and the path is go.

#### 5. Architecture

*Model configuration, small-model harness and design experiments.* Find the best model configuration with small experiments and freeze it on 14 Nov.

Timeline bars:

- 2026-10-24 to 2026-10-31: Config v0 and small-model harness
- 2026-11-01 to 2026-11-14: Design experiments (break 5 to 11 Nov: queued runs only)
- 2026-11-15 to 2026-11-21: Full-width runs and long-context plan in dry run 1
- 2026-11-22 to 2026-11-28: Dry run 2 and launch config

Milestones:

- 2026-10-27: Config v0 merged
- 2026-10-31: Experiment plan approved
- 2026-11-14: Config v1 frozen
- 2026-11-28: Launch config signed

**Week 1 — Config v0, harness and experiment plan**

- Students:
  - Do your diagnosis slice: 150 rows by 27 Oct, all rows by 31 Oct.
  - Write configs/config_v0.yaml with a default and a source for each setting.
  - Build the small-model harness. One seed must give the same loss each run.
  - Write experiments/plan_v0.md: each arm with metric, pass bar and GPU hours.
- TSAI Lab:
  - **HAVE** Week 1 uses your own GPU, Colab or Kaggle for tiny models.
  - **HAVE** Config v0 keeps size values open. TSAI gives them on 31 Oct.
- Depends on:
  - Team 4: KE2 output shape, 27 Oct
  - Team 8: trainer logger and alarms, 29 Oct
- Gate: Config v0 loads, the harness gives repeatable loss, and TSAI approves the experiment plan.

**Week 2 — Design experiments and config freeze**

- Students:
  - Run EXP-1, EXP-2 and EXP-3: learning rate, optimizer and expert count.
  - Queue the attention arms before 5 Nov. Break 5 to 11 Nov.
  - Run EXP-5 with Team 4 and report all results in bits per byte.
  - Write config_v1.yaml and decisions.md with a result for each setting, 14 Nov.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives GPU time on the shared H100 node.
  - **DECIDE** TSAI picks attention, expert count and optimizer, and signs config v1.
- Rotation task (students from other teams):
  - Run one arm of the learning-rate sweep on a tiny model.
  - Log the loss curve and the config in the team folder.
- Depends on:
  - Team 4: KE2 decision, 14 Nov
  - Team 3: frozen tokenizer, 14 Nov
- Gate: Config v1 is frozen on 14 Nov, each setting has a result, and TSAI signs it.

**Week 3 — Frozen config at full width in dry run 1**

- Students:
  - Retrain the best small arm with the frozen tokenizer and compare bits per byte.
  - Run config v1 at full width in dry run 1 and record loss.
  - Write docs/long_context_plan.md and test the extension step on a small model.
  - Check the reversible recompute and Memory Stream at full width with Team 6.
- TSAI Lab:
  - **DECIDE** TSAI sets the learning-rate schedule and context length per phase.
- Rotation task (students from other teams):
  - Repeat one experiment with a new seed.
  - Report if the result holds.
- Depends on:
  - Team 7: dry run 1 node, 15 Nov
  - Team 8: alarms live in dry run 1
- Gate: Dry run 1 ends on 21 Nov with a stable loss curve, and the long-context plan is tested.

**Week 4 — Dry run 2 and the launch config**

- Students:
  - Write configs/launch_config.yaml from config v1 before 24 Nov.
  - Watch dry run 2 and compare its loss with dry run 1.
  - After the planned failure, check that the restored loss equals the earlier loss.
  - Write the model-settings run-book page and present the launch config, 28 Nov.
- TSAI Lab:
  - **DECIDE** TSAI signs launch_config.yaml before dry run 2.
  - **DECIDE** No setting changes after 26 Nov without TSAI approval.
- Depends on:
  - Team 7: dry run 2, 24 Nov
  - Team 9: step-0 evaluation, 24 Nov
- Gate: The signed launch config equals the dry run 2 config, and the team gives a go vote.

#### 6. Performance and Kernels

*Speed and correctness of each part of the training step.* Time each part of the training step, prove that fast code is correct, and make it faster.

Timeline bars:

- 2026-10-24 to 2026-10-31: Timing harness and correctness framework
- 2026-11-01 to 2026-11-14: Node baseline and first speedups (break 5 to 11 Nov: no work)
- 2026-11-15 to 2026-11-21: Full-width profile in dry run 1
- 2026-11-22 to 2026-11-26: Kernel freeze for dry run 2
- 2026-11-15 to 2026-11-28: Speed work continues until launch

Milestones:

- 2026-10-31: Baseline table and five slowest parts
- 2026-11-14: Speed report for config v1
- 2026-11-23: Kernel set frozen for dry run 2
- 2026-11-28: Launch commit passes the correctness suite

**Week 1 — Timing harness, correctness framework and baseline**

- Students:
  - Do your diagnosis slice: 150 rows by 27 Oct, all rows by 31 Oct.
  - Write perfkit/timer.py and bench.py. Time each part of the training step.
  - Write correctness/framework.py and test pairs for KE, attention, MoE and reversible layers.
  - Write baseline_v0.md and slowest5_v0.md with the GPU name and commit ID.
- TSAI Lab:
  - **HAVE** Week 1 baselines run on your own GPU, Colab or Kaggle.
  - **HAVE** Target GPUs: AWS H100 and B200 nodes in us-east-1.
- Depends on:
  - Team 5: harness and small configs, 27 Oct
  - Team 4: KE2 decode reference, 28 Oct
- Gate: All correctness tests pass, a broken function fails, and repeat baseline runs agree within 5%.

**Week 2 — Node baseline and first speedups**

- Students:
  - Run the baseline on the shared H100 node and time each Team 5 arm.
  - Write one speedup, with a correctness test, for two of the slowest parts.
  - Time the config v1 shape. Write speed_v1.md with tokens per second and MFU.
  - Agree one MFU method with Team 8. Break 5 to 11 Nov.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives node time on the job queue for timing.
  - **DECIDE** TSAI picks the speedups that go into the trainer.
- Rotation task (students from other teams):
  - Profile one part of the training step.
  - Report the top three costs with the profiler trace.
- Depends on:
  - Team 7: job queue, 1 Nov
  - Team 5: experiment arms, 1 Nov
- Gate: Config v1 has a speed report, and each merged speedup has a passing correctness test.

**Week 3 — Full-width profile in dry run 1**

- Students:
  - Profile the full-width step in dry run 1 with torch.profiler and Nsight.
  - Time MoE dispatch, all-to-all and the full-size checkpoint save.
  - Test fused kernels for MoE dispatch and attention against the reference.
  - Write results/speed_dryrun1.md: step time per rank and slow ranks.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives node time for dry run 1 profiles.
  - **DECIDE** TSAI picks the kernel changes for dry run 2.
- Rotation task (students from other teams):
  - Write one correctness test for a fast kernel.
  - The test compares the reference and the fast output.
- Depends on:
  - Team 5: config v1, 14 Nov
  - Team 7: parallel layout, 15 Nov
- Gate: Dry run 1 has a full profile, and each kernel for dry run 2 passes its test.

**Week 4 — Kernel freeze and launch speed check**

- Students:
  - Freeze and tag the kernel set for dry run 2 on 23 Nov.
  - Measure tokens per second in dry run 2 and compare with dry run 1.
  - Run the correctness suite on the launch commit. Add it to the checklist.
  - Write docs/speed_backlog.md for speed work after launch.
- TSAI Lab:
  - **DECIDE** TSAI approves the launch commit of the trainer.
- Depends on:
  - Team 7: dry run 2, 24 Nov
  - Team 8: throughput alarm, 24 Nov
- Gate: The launch commit passes the correctness suite, and dry run 2 is as fast as dry run 1.

#### 7. Infrastructure

*Job queue, GPU health, checkpoints, parallel layout and dry runs.* Give the lab a GPU cluster that saves, restores and restarts the training run with no lost work.

Timeline bars:

- 2026-10-24 to 2026-10-31: Queue design, health check and checkpoint design
- 2026-11-01 to 2026-11-14: Node setup and restore test (break 5 to 11 Nov: no work)
- 2026-11-12 to 2026-11-21: Parallel layout and dry run 1
- 2026-11-24 to 2026-11-26: Dry run 2 with a planned failure
- 2026-11-27 to 2026-11-28: Launch readiness

Milestones:

- 2026-11-01: Job queue live on the shared node
- 2026-11-14: Restore test passes on the node
- 2026-11-21: Dry run 1 ends
- 2026-11-28: Go or no-go

**Week 1 — Queue, health check and checkpoint design**

- Students:
  - Do your diagnosis slice: 150 rows by 27 Oct, all rows by 31 Oct.
  - Write the queue design and test Slurm in Docker with two accounts.
  - Write launch.sh, gpu_health.sh and xid_scan.py with the NCCL all-reduce test.
  - Write the checkpoint design and ckpt.py. A CPU restore gives bit-equal loss.
- TSAI Lab:
  - **HAVE** The lab repository and the CAP chat groups are ready.
  - **HAVE** The first shared GPU node comes in week 2. Week 1 is design work.
- Depends on:
  - Team 5: config v0, 27 Oct
  - Team 8: health-check JSON format, 29 Oct
- Gate: TSAI approves both designs, and the Docker Slurm, health and CPU restore tests pass.

**Week 2 — Node setup, restore test and automatic restart**

- Students:
  - Run the health check and the NCCL baseline on the node, 1 Nov.
  - Install Slurm and the team accounts, then run the restore test.
  - Build automatic restart from the last COMPLETE checkpoint, 12 to 14 Nov.
  - Write parallel_layout_v0.md from config v1. Break 5 to 11 Nov.
- TSAI Lab:
  - **DECIDE** TSAI sets the GPU share for each team per day.
  - **CONFIRM** TSAI Lab gives S3 checkpoint storage and keeps the keys.
- Rotation task (students from other teams):
  - Run the restore test from a checkpoint and time it.
  - Write the steps and the time in the runbook.
- Depends on:
  - Team 6: save time at config v1 shape
  - Team 5: config v1, 14 Nov
- Gate: The restore test gives bit-equal loss, and automatic restart works on a killed job.

**Week 3 — Parallel layout and dry run 1**

- Students:
  - Test the parallel layouts on H100 and B200. Record tokens per second.
  - Run dry run 1 until 21 Nov with config v1 and first shards.
  - Set the checkpoint interval from the measured save and restore time.
  - Write docs/runbook.md for node failures, NCCL hangs and bad checkpoints.
- TSAI Lab:
  - **DECIDE** TSAI picks the parallel layout for the training run.
  - **CONFIRM** TSAI Lab reserves the nodes for the training run.
- Rotation task (students from other teams):
  - Watch dry run 1 for one shift.
  - Log each failure and each restart.
- Depends on:
  - Team 8: alarms and dashboards live, 15 Nov
  - Team 6: full-size save time, 18 Nov
- Gate: Dry run 1 ends on 21 Nov with no lost steps and only COMPLETE checkpoint restarts.

**Week 4 — Dry run 2 with a planned failure**

- Students:
  - Write failure_plan.md for dry run 2: kill one rank, then one node.
  - Run dry run 2, make the planned failure and record the restart time.
  - Run dcgmi diag and the NCCL test on each training node, 27 Nov.
  - Present the dry run 2 report at the go or no-go meeting.
- TSAI Lab:
  - **DECIDE** TSAI makes the go or no-go call on 28 Nov.
- Depends on:
  - Team 6: kernel set frozen, 23 Nov
  - Team 8: on-call rota, 27 Nov
- Gate: Dry run 2 restarts on its own with equal loss, and each node passes the health check.

#### 8. Monitoring and Records

*Metrics, alarms, dashboards, per-token records and on-call.* Find each training problem in time and keep a full record of the run.

Timeline bars:

- 2026-10-24 to 2026-10-31: Metric list, trainer logger and alarms
- 2026-11-01 to 2026-11-14: ClickHouse rollups and private Grafana (break 5 to 11 Nov: no work)
- 2026-11-15 to 2026-11-21: Live monitoring and per-token records in dry run 1
- 2026-11-22 to 2026-11-28: Alert tests in dry run 2 and on-call rota

Milestones:

- 2026-10-31: Alarms run inside the harness
- 2026-11-14: Private Grafana live
- 2026-11-21: Public read-only Grafana live
- 2026-11-28: On-call rota starts

**Week 1 — Metric list, logger, alarms and diagnosis tracker**

- Students:
  - Do your diagnosis slice: 150 rows by 27 Oct, all rows by 31 Oct.
  - Write metric_list_v0.yaml and metrics_writer.py, which flushes every 10 steps.
  - Write alarms.py and test it on 3,000 synthetic values.
  - Build the diagnosis tracker from a private roster file, with no emails in git.
- TSAI Lab:
  - **HAVE** Week 1 uses your own machine for ClickHouse.
  - **HAVE** The pack assignment table on this page is the tracker input.
- Depends on:
  - Team 5: harness metric names, 27 Oct
  - Team 6: system metrics and MFU method
- Gate: The alarm tests pass, the logger runs in the Team 5 harness, and the tracker works.

**Week 2 — ClickHouse, rollups and private Grafana**

- Students:
  - Install ClickHouse and load the Team 5 experiment runs into metrics_raw.
  - Build rollup tables at 10, 100, 1,000 and 10,000 steps.
  - Build the private Grafana, with KE2 metrics, after the 5 to 11 Nov break.
  - Connect the node logs to the ClickHouse loader. Write per_token_records.md.
- TSAI Lab:
  - **DECIDE** TSAI sets the eval interval and the public panel list.
- Rotation task (students from other teams):
  - Add one metric to the trainer logger.
  - Add its panel to the private dashboard.
- Depends on:
  - Team 7: node logs, 1 Nov
  - Team 4: KE2 metric names, 2 Nov
- Gate: Experiment runs show in the private Grafana, and ClickHouse rows equal the JSONL lines.

**Week 3 — Live monitoring of dry run 1**

- Students:
  - Monitor dry run 1 live. Each metric reaches Grafana within one minute.
  - Write alarms_dryrun1.md and store per-token records as Parquet.
  - Build the public read-only Grafana with the panels that TSAI approves.
  - Write docs/alert_tests.md for dry run 2.
- TSAI Lab:
  - **DECIDE** TSAI approves the public panels and the record keep time.
  - **CONFIRM** TSAI Lab gives the host for the public Grafana.
- Rotation task (students from other teams):
  - Test two alerts with an injected fault.
  - Record the time to alert.
- Depends on:
  - Team 7: dry run 1 node, 15 Nov
  - Team 9: checkpoint scores, 21 Nov
- Gate: Dry run 1 has full metrics with no missing steps, and the public Grafana is live.

**Week 4 — Alert tests and on-call rota**

- Students:
  - Inject a NaN, a loss spike and a stopped rank in dry run 2.
  - Measure fault-to-page time and check alarm state after the restore.
  - Connect pages to the alert channel that the on-call team watches.
  - Write the on-call rota and run-book, with one Team 7 member per shift.
- TSAI Lab:
  - **DECIDE** TSAI approves the rota and the page channel.
- Depends on:
  - Team 7: dry run 2, 24 Nov
  - Team 11: public dashboard text, 27 Nov
- Gate: Each injected fault fires the correct alarm, and the approved rota starts on 28 Nov.

#### 9. Evaluation

*Benchmarks, evaluation harness, benchmark fingerprints and checkpoint evaluation.* Measure the model with fair benchmarks and keep every benchmark item out of the training data.

Timeline bars:

- 2026-10-24 to 2026-10-31: Benchmark list v1 and harness on open models
- 2026-10-24 to 2026-11-04: Benchmark fingerprints and decontamination check
- 2026-11-01 to 2026-11-14: Eval firewall and canary strings (break 5 to 11 Nov, no work)
- 2026-11-12 to 2026-11-21: Automatic evaluation of checkpoints
- 2026-11-22 to 2026-11-28: Eval at step 0 of dry run 2 and run schedule

Milestones:

- 2026-10-28: Fingerprints v1 to the data teams
- 2026-10-31: Harness agrees with published scores
- 2026-11-21: Automatic checkpoint evaluation live
- 2026-11-24: Step-0 eval of dry run 2

**Week 1 — Diagnosis, benchmark list and fingerprints v1**

- Students:
  - Do your diagnosis slice: 150 rows by 27 Oct, all rows by 31 Oct.
  - Write benchmarks_v1.md with the licence and item count of each benchmark.
  - Write build_fingerprints.py and decontam_check.py with tests. Give them to the data teams.
  - Run the harness on three open models and make eval_mini_v0.json.
- TSAI Lab:
  - **HAVE** Week 1 harness runs use your own GPU, Colab or Kaggle.
  - **HAVE** Indic list: the 10 languages of our packs.
- Depends on:
  - TSAI Lab: diagnosis sample packs, 24 Oct
  - Team 10: special-token draft, 28 Oct
- Gate: The data teams have fingerprints v1, and harness scores are within 2 points of published scores.

**Week 2 — Fingerprints v2 and the eval firewall**

- Students:
  - Rebuild the fingerprints with the spec v1 normalize function. Share the sha256.
  - Write the eval firewall rule and the storage access list.
  - Add canary strings and run canary_scan.py on the tokenizer sample.
  - Run eval-mini on Team 4 and Team 5 checkpoints after the break.
- TSAI Lab:
  - **CONFIRM** TSAI Lab makes the separate eval storage folder.
  - **DECIDE** TSAI picks the eval-mini scores for the config freeze.
- Rotation task (students from other teams):
  - Run one benchmark on one open model.
  - Compare the score with the published score.
- Depends on:
  - Team 1: spec v1, 31 Oct
  - Team 4 and Team 5: checkpoints, 12 Nov
- Gate: Fingerprints v2 match spec v1, and the canary scan finds no canary in the tokenizer sample.

**Week 3 — Automatic evaluation during dry run 1**

- Students:
  - Write eval_watcher.py. It scores each new checkpoint into ClickHouse.
  - Run decontamination of the first training shards with fingerprints v2.
  - Check 50 flagged documents by hand and make the Grafana eval panel.
  - Write eval_schedule.md: the benchmarks for each step of the run.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives a GPU slot for the eval watcher.
  - **DECIDE** TSAI approves the eval schedule for the training run.
- Rotation task (students from other teams):
  - Run the decontamination check on 2 shards.
  - Report each hit with the benchmark item.
- Depends on:
  - Team 7: dry run 1 checkpoints
  - Team 8: ClickHouse eval tables, 18 Nov
- Gate: The watcher scores each dry run 1 checkpoint, and the first shards have zero fingerprint hits.

**Week 4 — Step-0 eval of dry run 2 and launch readiness**

- Students:
  - Run the step-0 eval of dry run 2. Scores must be near random.
  - Check that the watcher scores the restored checkpoint after the planned failure.
  - Give Team 11 the benchmark names and the score format.
  - Write the on-call note for eval failures.
- TSAI Lab:
  - **DECIDE** TSAI approves each public score.
  - **DECIDE** TSAI gives go or no-go for evaluation, 28 Nov.
- Depends on:
  - Team 7: dry run 2, 24 Nov
  - Team 11: dashboard text format, 25 Nov
- Gate: Step-0 scores are near random, and the watcher survives the planned failure of dry run 2.

#### 10. Post-training and Agentic

*SFT and RL strategy, special tokens, chat format, SFT and RL data, SWE environments.* Prepare the special tokens, chat format, data and RL loop for an agentic coding model.

Timeline bars:

- 2026-10-24 to 2026-10-31: Licence table and strategy v0
- 2026-10-24 to 2026-10-31: Special-token list and chat format
- 2026-11-01 to 2026-11-21: SFT data v1 (break 5 to 11 Nov, no work)
- 2026-11-01 to 2026-11-21: RL data v1 with verifiers and SWE environments (break 5 to 11 Nov)
- 2026-11-15 to 2026-11-28: RL loop on a small model

Milestones:

- 2026-10-28: Special-token list draft
- 2026-10-31: Special-token list final
- 2026-11-21: SFT data v1 and RL data v1
- 2026-11-28: RL loop runs on a small model

**Week 1 — Diagnosis, licences and special tokens**

- Students:
  - Do your diagnosis slice: 150 rows by 27 Oct, all rows by 31 Oct.
  - Write licence_table.md with the licence, source and decision for each set.
  - Write special_tokens_v0.md by 28 Oct and the final list by 31 Oct.
  - Write chat_format_v0.md and render.py. Run one SWE task end to end.
- TSAI Lab:
  - **HAVE** Week 1 uses your own machine with Docker.
  - **HAVE** Rules: commercial-use data only; 256 reserved tokens; effort budgets 1k, 4k, 16k, 64k.
- Depends on:
  - Team 3: tokenizer training starts 1 Nov
  - Team 9: fingerprints v1, 28 Oct
- Gate: The final special-token list is with Team 3 on 31 Oct, and each set has a licence decision.

**Week 2 — SFT data v1 start and SWE environments**

- Students:
  - Convert the approved SFT sets to the chat format and write the manifest.
  - Run decontam_check.py on every SFT and RL set and record removals.
  - Build 20 SWE environments in Docker after the break. Each gold patch passes.
  - Write code tasks with hidden tests and a verifier for RL data v1.
- TSAI Lab:
  - **CONFIRM** TSAI Lab downloads the approved sets and publishes them as packs.
  - **DECIDE** TSAI sets the share of each area in the SFT mix.
- Rotation task (students from other teams):
  - Check the licences of 5 SFT or RL data sets.
  - Add them to the licence table.
- Depends on:
  - Team 3: frozen tokenizer, 14 Nov
  - Team 9: fingerprints v2, 4 Nov
- Gate: render.py passes all tests with the frozen tokenizer, and twenty SWE environments pass the gold check.

**Week 3 — SFT data v1, RL data v1 and verifiers**

- Students:
  - Finish SFT data v1 with loss masks and RL data v1 with verifiers.
  - Test each verifier with one correct answer and one wrong answer.
  - Run SFT on a small open model and check that the loss falls.
  - Test the effort levels on 100 fixed prompts.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives GPU time for the small SFT run.
  - **DECIDE** TSAI approves SFT and RL data v1 at Gate 3.
- Rotation task (students from other teams):
  - Run one software-engineering task end to end in Docker.
  - Record the verifier result.
- Depends on:
  - Team 5: small MoE model, 18 Nov
  - Team 9: eval-mini, 15 Nov
- Gate: SFT and RL data v1 pass the decontamination check, and every verifier passes its two tests.

**Week 4 — RL loop on a small model**

- Students:
  - Run the RL loop on the small MoE model with frozen router weights.
  - Check that the router checksum stays the same after RL.
  - Read 50 high-reward answers and record each cheat in reward_hacking.md.
  - Score the RL model with Team 9. Give Team 11 the licence section.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives GPU time for the RL loop.
  - **DECIDE** TSAI picks the RL algorithm and gives go or no-go.
- Depends on:
  - Team 9: RL model scores, 27 Nov
  - Team 11: model card outline, 22 Nov
- Gate: The RL loop raises the reward with a frozen router, and Team 11 has the licence section.

#### 11. Release

*Model card, report, weekly public updates, public dashboard text and launch announcement.* Tell the public what the lab does, in clear and correct words that TSAI approves.

Timeline bars:

- 2026-10-24 to 2026-10-31: Style guide, public and internal list, templates
- 2026-10-24 to 2026-11-04: Model card and report outlines
- 2026-10-24 to 2026-11-28: Weekly public updates (no update 5 to 11 Nov)
- 2026-11-12 to 2026-11-21: Public dashboard text
- 2026-11-15 to 2026-11-28: Launch announcement

Milestones:

- 2026-10-31: First public update to TSAI
- 2026-11-21: Dashboard text to Team 8
- 2026-11-28: Launch announcement approved

**Week 1 — Diagnosis, rules and the first update**

- Students:
  - Do your diagnosis slice: 150 rows by 27 Oct, all rows by 31 Oct.
  - Write public_internal.md and style_guide.md with check_style.py.
  - Write consent_form.md for names in public.
  - Write the update template and the first update, with one sentence per team.
- TSAI Lab:
  - **HAVE** Nothing is public without TSAI approval.
  - **HAVE** The CAP chat group G11 is the review place.
- Depends on:
  - All teams: one sentence each, 28 Oct
  - Team 9: benchmark names, 31 Oct
- Gate: The first update has 0 style errors, and TSAI has it on 31 Oct.

**Week 2 — Outlines and the second update**

- Students:
  - Write model_card_outline.md with an owner team for each section.
  - Write report_outline.md with the V4 run report as the model.
  - Write the week 2 update. No update from 5 to 11 Nov.
  - Write the Gate 2 update draft after the break.
- TSAI Lab:
  - **DECIDE** TSAI approves the week 2 update and public Gate 2 results.
- Rotation task (students from other teams):
  - Write the weekly-update paragraph for your core team.
  - Use plain facts and no numbers that TSAI has not approved.
- Depends on:
  - Team 3: tokenizer results, 14 Nov
  - Team 4: KE2 decision, 14 Nov
- Gate: Both outlines have an owner team for each section, and TSAI has the Gate 2 update.

**Week 3 — Public dashboard text**

- Students:
  - Write dashboard_text.md: one plain sentence for each public Grafana panel.
  - Remove every internal value and write a public glossary.
  - Write the Gate 3 update on dry run 1 and the first shards.
  - Start launch_announcement_v0.md.
- TSAI Lab:
  - **DECIDE** TSAI approves the dashboard text and the Gate 3 update.
- Rotation task (students from other teams):
  - Review one section of the model card draft.
  - Mark each claim that has no evidence.
- Depends on:
  - Team 8: public panel list, 17 Nov
  - Team 7: dry run 1 summary, 21 Nov
- Gate: Team 8 has the approved dashboard text, and the launch announcement draft exists.

**Week 4 — Launch announcement and model card draft**

- Students:
  - Finish the launch announcement with the goal, the teams and the dashboard link.
  - Fill the known model card sections: data, licences and evaluation plan.
  - Run check_style.py on every text. The result must be 0 errors.
  - Write the 28 Nov update and a source list for each fact.
- TSAI Lab:
  - **DECIDE** TSAI approves each published text word by word.
  - **CONFIRM** TSAI sets the announcement time on 28 Nov.
- Depends on:
  - Team 10: licence section, 27 Nov
  - Team 9: public score format, 26 Nov
- Gate: TSAI approves the launch announcement, and each model card section has an owner and a status.

#### 12. Quantization and Serving

*Quantization tools, score drop at each bit width, automatic quantize flow, public demo and ollama converter.* Make each checkpoint small and fast to run, and measure how much score it loses.

Timeline bars:

- 2026-10-24 to 2026-10-31: Tools and bf16 baseline on an open MoE
- 2026-10-27 to 2026-11-04: Bit width against score drop table
- 2026-11-01 to 2026-11-14: Unsupported parts: KE2, Memory Stream, attention (break 5 to 11 Nov)
- 2026-11-12 to 2026-11-26: Automatic flow: checkpoint to quantized model to benchmark
- 2026-11-15 to 2026-11-28: Serving plan, public demo and ollama converter

Milestones:

- 2026-11-04: Score drop table
- 2026-11-21: Flow tested on a dry run 1 checkpoint
- 2026-11-28: Flow tested on dry run 2 and serving plan to TSAI

**Week 1 — Diagnosis, tools and the bf16 baseline**

- Students:
  - Do your diagnosis slice: 150 rows by 27 Oct, all rows by 31 Oct.
  - Write tools.md with MoE support and licence for each quantization tool.
  - Run the bf16 baseline of one small open MoE on eval-mini.
  - Quantize it at five or more bit widths and record each in runs.csv.
- TSAI Lab:
  - **HAVE** Week 1 uses your own GPU, Colab or Kaggle.
  - **HAVE** Test models: OLMoE-1B-7B and Qwen3-30B-A3B.
- Depends on:
  - Team 9: eval_mini_v0.json, 28 Oct
  - Team 10: chat_format_v0.md, 31 Oct
- Gate: runs.csv has the bf16 baseline and three or more quantized formats, each with time and machine.

**Week 2 — Score drop table and unsupported parts**

- Students:
  - Finish bitwidth_vs_drop.md and recommend one format for the demo.
  - Write quantize_eval.py v0 with a sha256 for each output file.
  - Write unsupported_parts.md: KE2 output, Memory Stream, attention and tokenizer.
  - After the break, add a plan for each custom part of config v1.
- TSAI Lab:
  - **DECIDE** TSAI picks the bit widths for the automatic flow.
- Rotation task (students from other teams):
  - Quantize one small open model to 4-bit.
  - Measure the score drop on the small benchmark subset.
- Depends on:
  - Team 5: frozen configuration, 14 Nov
  - Team 3: frozen tokenizer, 14 Nov
- Gate: bitwidth_vs_drop.md has five or more formats, and each custom part has a plan.

**Week 3 — Automatic flow on dry run 1**

- Students:
  - Extend quantize_eval.py: checkpoint in, quantized model and scores out, no manual step.
  - Add the new tokenizer to the GGUF converter and test round trips.
  - Run the flow on a dry run 1 checkpoint and test 100 fixed prompts.
  - Start serving_plan.md: demo, server, format and expected speed.
- TSAI Lab:
  - **CONFIRM** TSAI Lab gives GPU time for the flow.
  - **DECIDE** TSAI picks the serving stack for the public demo.
- Rotation task (students from other teams):
  - Run the quantize-and-evaluate flow on one checkpoint.
  - Record the time and the cost.
- Depends on:
  - Team 7: dry run 1 checkpoint, 19 Nov
  - Team 9: eval watcher score format
- Gate: The flow quantizes and scores one dry run 1 checkpoint, and the tokenizer round trip passes.

**Week 4 — Flow on dry run 2, serving plan and ollama converter**

- Students:
  - Run the flow on a dry run 2 checkpoint and match the watcher scores.
  - Write ollama_converter.md and a Modelfile with the Team 10 chat template.
  - Finish serving_plan.md with a ten-user load test of the demo server.
  - Write the on-call note and one paragraph for Team 11.
- TSAI Lab:
  - **DECIDE** TSAI approves the serving plan.
  - **DECIDE** TSAI gives go or no-go for the flow, 28 Nov.
- Depends on:
  - Team 7: dry run 2 checkpoint, 26 Nov
  - Team 10: final chat template, 26 Nov
- Gate: The flow works on a dry run 2 checkpoint, and TSAI has the serving plan.

### Sample packs (week 1)

Pack download links expire Fri 30 Oct, 17:00 IST. Each pack is about 500 documents; two or three students read each one.

| Pack | Language | Source | Readers |
|---|---|---|---|
| `cadj-commits` | Code | bigcode/commitpackft | 3 |
| `cadj-notebooks` | Code | HuggingFaceTB/issues-kaggle-notebooks | 3 |
| `cadj-stackexchange` | English | bigcode/starcoder2data-extras | 3 |
| `code-cpp` | Code | bigcode/starcoderdata | 3 |
| `code-go` | Code | bigcode/starcoderdata | 3 |
| `code-java` | Code | bigcode/starcoderdata | 3 |
| `code-js-ts` | Code | bigcode/starcoderdata | 3 |
| `code-python` | Code | bigcode/starcoderdata | 3 |
| `code-rust` | Code | bigcode/starcoderdata | 3 |
| `en-dclm` | English | mlfoundations/dclm-baseline-1.0-parquet | 3 |
| `en-fineweb-edu` | English | HuggingFaceFW/fineweb-edu | 3 |
| `en-nemotroncc` | English | nvidia/Nemotron-CC (v1, data.commoncrawl.org contrib/Nemotron) | 2 |
| `en-nemotroncc-synth` | English | nvidia/Nemotron-CC (v1, data.commoncrawl.org contrib/Nemotron) | 2 |
| `ind-bn-fw2` | Bengali | HuggingFaceFW/fineweb-2 | 2 |
| `ind-bn-sangraha-v` | Bengali | ai4bharat/sangraha | 3 |
| `ind-gu-fw2` | Gujarati | HuggingFaceFW/fineweb-2 | 2 |
| `ind-hi-fw2` | Hindi | HuggingFaceFW/fineweb-2 | 2 |
| `ind-hi-hplt` | Hindi | HPLT 3.0 (data.hplt-project.org/three/sorted; HF HPLT/HPLT3.0 is README-only) | 3 |
| `ind-hi-indiccorp` | Hindi | ai4bharat/IndicCorpV2 | 3 |
| `ind-hi-sangraha-s` | Hindi | ai4bharat/sangraha | 3 |
| `ind-hi-sangraha-v` | Hindi | ai4bharat/sangraha | 2 |
| `ind-kn-fw2` | Kannada | HuggingFaceFW/fineweb-2 | 2 |
| `ind-kn-sangraha-s` | Kannada | ai4bharat/sangraha | 2 |
| `ind-kn-sangraha-v` | Kannada | ai4bharat/sangraha | 2 |
| `ind-ml-fw2` | Malayalam | HuggingFaceFW/fineweb-2 | 2 |
| `ind-mr-fw2` | Marathi | HuggingFaceFW/fineweb-2 | 3 |
| `ind-mr-sangraha-v` | Marathi | ai4bharat/sangraha | 2 |
| `ind-or-fw2` | Odia | HuggingFaceFW/fineweb-2 | 2 |
| `ind-sa-fw2` | Sanskrit | HuggingFaceFW/fineweb-2 | 2 |
| `ind-ta-fw2` | Tamil | HuggingFaceFW/fineweb-2 | 2 |
| `ind-ta-sangraha-s` | Tamil | ai4bharat/sangraha | 2 |
| `ind-ta-sangraha-v` | Tamil | ai4bharat/sangraha | 2 |
| `ind-te-fw2` | Telugu | HuggingFaceFW/fineweb-2 | 3 |
| `ind-te-sangraha-s` | Telugu | ai4bharat/sangraha | 2 |
| `ind-te-sangraha-v` | Telugu | ai4bharat/sangraha | 2 |
| `math-finemath` | English | HuggingFaceTB/finemath | 2 |
| `math-megamath` | English | LLM360/MegaMath | 2 |
| `sci-finepdfs` | English | HuggingFaceFW/finepdfs | 1 |

Totals: 92 students with a pack, 38 packs, 45 on the waitlist. Sheet last updated 26 Oct 2026.