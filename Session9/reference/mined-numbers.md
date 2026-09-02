# Session 9 — numbers mined from the widgets

Everything below was read out of the widget source under `widgets/`, not out of the lesson prose.
Where a widget and the lesson disagree, the disagreement is noted — the widget is the executable
version and is what an implementation gets measured against.

---

## widget_1_head.html — the output head, row by row

Toy head: `D = 8`, `NV = 12`. Vocabulary
`['mat','floor','table','roof','cat','dog','sat','ran','the','a','quickly','again']`.

The weight matrix is generated deterministically so the page is identical on every load:
LCG seeded `s = 77001`, `s = (1103515245·s + 12345) mod 2³²`, each entry
`round((s/2³² · 2 − 1) · 100)/100`.

Four fixed contexts, each an 8-vector:

| Context | h |
| ------- | - |
| `the cat sat on the` | [0.92, −0.31, 0.55, 0.18, −0.44, 0.71, 0.26, −0.12] |
| `the dog ran across the` | [0.41, 0.63, −0.22, 0.77, 0.15, −0.36, 0.58, 0.29] |
| `she put it on the` | [0.66, −0.08, 0.34, −0.51, 0.62, 0.23, −0.40, 0.47] |
| `it happened` | [−0.27, 0.49, 0.12, 0.35, −0.58, −0.19, 0.81, 0.06] |

Real-scale figures the widget prints (these are the session's anchor numbers):

| Quantity | Value |
| -------- | ----- |
| V | 131,072 |
| d_model | 4,096 |
| head parameters, and multiply-adds per token | 536,870,912 |
| Session 7 trainable input side | 33,554,432 |
| ratio | 16.0× |

Toy cost for contrast: 12 dot products of length 8 = 96 multiply-adds per token.

## widget_2_cross_entropy.html — the gradient, checked against finite differences

Five tokens `['mat','floor','table','roof','chair']`, correct index `Y = 0` (mat).
Starting logits `Z0 = [2.4, 0.8, 3.1, −0.5, 1.2]`.

| Slider | min | max | step | default |
| ------ | --- | --- | ---- | ------- |
| `z[i]`, one per token | −600 | 600 | 1 | `round(Z0[i]·100)` — value is /100, so range is −6.0 … +6.0 |

- Finite-difference step `HS = 1e-4`, **central** difference: `(L(z+h·e_i) − L(z−h·e_i)) / 2h`.
- Analytic gradient: `softmax(z)_i − onehot(Y)_i`.
- The widget reports the worst analytic-vs-numeric disagreement and the sum of the analytic
  gradient, which is what makes the update a redistribution rather than a shift.
- Loss is computed the numerically stable way (`−(z_y − m − log Σ exp(z−m))`), not as `−log(p)`.
- Metrics panel shows the loss in nats, in bits (`L / ln 2`), and the single-token perplexity
  `exp(L)`.

Note the lesson's worked example in §4–5 uses a *different* five logits
([2.0, 1.0, 0.5, −1.0, −1.5], correct = Chennai). The widget's numbers are its own.

## widget_3_targets.html — where the targets come from

Fixed sequence of 14 positions, two packed documents plus padding:

    ['the','model','reads','the','token','<eos>',
     'rain','fell','on','the','city','<eos>','<pad>','<pad>']

Document ids: `[0,0,0,0,0,0, 1,1,1,1,1,1,1,1]` — note positions 12 and 13, the two `<pad>`
entries, are labelled document 1.

Four switches, each a way to get the targets wrong: shift amount, count padding, allow a target to
cross the document boundary, and divide by the wrong denominator.

- Add-α smoothing constant `AL = 0.001` in the toy per-token loss
  `−log((count + α) / (total + α·V))`, where the "model" is a count table over the sequence itself.
- Correct reference setting: `shift = 1`, padding masked.
- Exclusion reasons the widget renders per position: `no target`, `pad input`, `pad target`,
  `crosses documents`.

## widget_4_logit_drift.html — log Z drift and the three fixes

Toy classifier: `V = 20`, `D = 12`, `N = 48` examples (`NU = 32` unique hidden states, the other
16 repeat an earlier state under a different label, which gives the data an irreducible entropy
floor the widget computes and shows).

| Constant | Value |
| -------- | ----- |
| `STEPS` | 500 |
| `LR` | 5 |
| `LAM` (z-loss λ) | 1e-3 |
| `CAP` (soft-cap c) | 30 |
| data seed | 7 |
| init seed | 11 |

Four modes from one identical initialisation: `baseline`, `zloss`, `softcap`, `center`.
Labels: `plain cross-entropy`, `+ z-loss, λ = 1e-3`, `+ soft-cap, c = 30`, `+ output centering`.

Mechanism details worth carrying:

- The bias is calibrated at step 0 so **log Z reads exactly 0 at step 0** in every mode. Drift is
  therefore measured from a common zero.
- z-loss enters the gradient as `g += 2·λ·logZ·p_v`, not as a separate backward pass.
- Soft-cap multiplies the gradient by `1 − tanh²(z/c)`.
- Centering subtracts the column mean from W **and** the mean of b, once per step, before the
  gradient is taken.

| Slider | min | max | step | default |
| ------ | --- | --- | ---- | ------- |
| `step` | 0 | 500 | 1 | 500 (end of run) |

Final numbers the lesson quotes from this widget's run: centering drives the mean logit to
**−5.07e-16** and leaves log Z at **52.17**, above the plain run's **46.42**; z-loss pins log Z to
**0.07**. The lesson explicitly corrects itself here — centering pins the *mean logit*, not log Z.

Soft-cap reference table (c = 30): logit 10 → 9.6, logit 60 → 28.9, logit 600 → 30.0.

## widget_5_mtp.html — multi-token prediction, four heads

Toy vocabulary of 12 words, `D = 8`, `NH = 4` heads, 10 fixed training sentences.
Trunk seed `sd = 20260916`, xorshift RNG; trunk is a tanh recurrence over `EMB`, `Wh`, `We`
(scales 0.9, 0.55, 0.85).

Each head is fitted in-page by full-batch gradient descent: **600 epochs, learning rate 3**.
Head k is trained on pairs `(trunk(s, t), s[t+k])`, so head 4 has fewer training pairs per sentence
than head 1 — part of why its loss sits higher.

| Slider | min | max | step | default |
| ------ | --- | --- | ---- | ------- |
| `pos` (position t) | 0 | 5 | 1 | 3 |

## widget_6_preference.html — RM / DPO / GRPO, and the log Z cancellation

One prompt, two completions, four candidate tokens per position with fixed reference logits.

| Constant | Value |
| -------- | ----- |
| winner tokens | `['it','follows','the','slope','downhill']` (5 positions) |
| loser tokens | `['it','is','a','thing']` (4 positions) |
| `TAKW` (winner's chosen indices) | [0, 0, 1, 0, 2] |
| `TAKL` (loser's chosen indices) | [0, 1, 2, 3] |
| `BETA` | 0.10 |
| `STEP` | 6 — the policy shift applied to the winner (+0.10·STEP) and loser (−0.08·STEP) |

| Slider | min | max | step | default | maps to |
| ------ | --- | --- | ---- | ------- | ------- |
| `rw` | −300 | 300 | 5 | 140 | reward on chosen, /100 → **1.40** |
| `rl` | −300 | 300 | 5 | 30 | reward on rejected, /100 → **0.30** |
| `lz` | −450 | 130 | 1 | 0 | log Z of the prompt, /100 → −4.50 … +1.30 |
| `g` | 4 | 12 | 1 | 8 | completions sampled per prompt (GRPO tab) |

The cancellation is made **exact, not approximate**: rewards are carried in two-term
(compensated / double-double) form via `twoSum` and `ddSub`, so the Bradley-Terry loss with log Z
matches the DPO loss to every printed digit. The widget also sweeps every log Z stop and reports
the worst difference across the whole slider range. This is the widget the lesson tells you to
spend the most time on.

## widget_7_logits_tensor.html — the logits tensor, drawn to scale

| Constant | Value |
| -------- | ----- |
| `V` | 131,072 |
| `D` | 4,096 |
| `A0` | 78,200 (px² allotted to the largest box) |
| `MAX` | 512 GiB, the area reference |
| card the verdict tests against | 80 GiB |

| Slider | min | max | default | maps to |
| ------ | --- | --- | ------- | ------- |
| `b` | 0 | 3 | 3 | batch = 2^b, so 1 … 8, **default 8** |
| `t` | 13 | 18 | 13 | context = 2^t, so 8,192 … 262,144, **default 8,192** |

Both tensors are counted at bf16 (2 bytes). Ratio printed as `V/D` = **32.0×**.
Verdict tests `logits + gradient ≤ 80 GiB`. Reproduces the lesson's table exactly:

| B | T | hidden | logits | logits + backward | fits 80 GiB? |
| - | - | ------ | ------ | ----------------- | ------------ |
| 8 | 8,192 | 0.5 GiB | 16 GiB | 32 GiB | yes |
| 4 | 32,768 | 1.0 GiB | 32 GiB | 64 GiB | yes |
| 1 | 262,144 | 2.0 GiB | 64 GiB | 128 GiB | no |

## widget_8_chunking.html — chunked cross-entropy

| Constant | Value |
| -------- | ----- |
| `N` (tokens) | 65,536 |
| `V` | 131,072 |
| per-token losses | deterministic Lehmer RNG, seed 2463534, `s = (s·16807) mod 2147483647`, loss = `(512 + s mod 5632)/1024` so losses lie in [0.5, 6.0) |

| Slider | min | max | step | default | maps to |
| ------ | --- | --- | ---- | ------- | ------- |
| `chunk` | 0 | 6 | 1 | 0 | chunk size = `N >> value`, so 65,536 down to **1,024** |

- Peak logits memory = `chunk × V × 2` bytes. Full pass: **16 GiB**. At chunk 1,024: **256 MiB**.
  Reduction **64×** — the lesson's figure.
- Both means are printed to **10 decimal places** with their difference, to show the chunked route
  is bit-for-bit identical, not merely close.

## widget_9_sft_mask.html — the SFT loss mask

Fixed 13-token conversation. `p = 1` marks a prompt token, `p = 0` a completion token; the first
token has no target.

| Token | per-token loss | prompt? |
| ----- | -------------- | ------- |
| `<\|user\|>` | (no target) | yes |
| `What` | 3.25 | yes |
| `is` | 1.5 | yes |
| `the` | 0.75 | yes |
| `capital` | 2 | yes |
| `of` | 0.625 | yes |
| `India` | 1.75 | yes |
| `?` | 1.25 | yes |
| `<\|assistant\|>` | 0.25 | yes |
| `New` | 2.5 | no |
| `Delhi` | 0.375 | no |
| `.` | 0.625 | no |
| `<\|end\|>` | 0.125 | no |

Masked (completion only): 4 contributing targets. Unmasked: 12 of 12 targets, and the widget
reports what share of the total loss came from tokens the *user* wrote — the number that makes the
point.

## widget_10_preference_gap.html — Bradley-Terry, and why only the gap matters

| Slider | min | max | step | default |
| ------ | --- | --- | ---- | ------- |
| `rw` reward on the winner | −4 | 4 | 0.25 | 1.5 |
| `rl` reward on the loser | −4 | 4 | 0.25 | 0.25 |
| `c` constant added to both | −3 | 3 | 0.25 | 0 |

Loss computed as a numerically stable softplus:
`bt(g) = g ≥ 0 ? log1p(exp(−g)) : −g + log1p(exp(g))` — not `−log(sigmoid(g))`, which would lose
precision at large |g|. The widget prints the loss at constant 0 alongside the shifted loss and
their difference, which stays 0 across the whole slider range.

Lesson's reference table for the sigmoid: gap 2.0 → 88%, gap 0.5 → 62%, gap 0.0 → 50%.

## widget_11_four_models.html — how many models each method keeps resident

Model size `P = 7e9`. Per-model memory:

| Role | bytes per parameter | GiB at 7B | trained? |
| ---- | ------------------- | --------- | -------- |
| Policy | 16 | 104.3 | trained |
| Reference | 2 | 13.0 | frozen |
| Reward | 2 | 13.0 | frozen |
| Value | 16 | 104.3 | trained |
| **all four** | | **234.7** | |

The 16 bytes for a trained model is Session 10's per-weight training state; the 2 bytes for a
frozen one is bf16 weights only.

| Mode | keeps | resident | saved |
| ---- | ----- | -------- | ----- |
| PPO | policy, reference, reward, value | 234.7 GiB | — |
| DPO | policy, reference | 117.3 GiB | 117.3 GiB |
| GRPO | policy, reference, reward | 130.4 GiB | 104.3 GiB |

The widget notes RLVR brings GRPO down to two as well, by putting a verifier in place of the reward
model.

## widget_12_loss_map.html — one KL with a swappable first argument

Model distribution q is fixed, from logits `z = [2, 1, 0.5, −1, −1.5]` over
`['Delhi','Mumbai','Chennai','banana','runs']` — the same logits as lesson §4.

Four first arguments:

| Mode | p | Formula shown |
| ---- | - | ------------- |
| Next-token cross-entropy | [1, 0, 0, 0, 0] | `L = −log q(Delhi)` |
| Label smoothing | [0.9, 0.025, 0.025, 0.025, 0.025] | ε = 0.1 |
| Distillation | [0.55, 0.25, 0.12, 0.05, 0.03] | `KL(teacher ‖ student)` |
| The RLHF KL penalty | [0.5, 0.22, 0.16, 0.07, 0.05] | `KL(reference ‖ policy)`, scaled by β |

For each it prints H(p), H(p,q) and KL, and flags the one-hot case where H(p) = 0 makes
cross-entropy and KL the same number.

**Stale cross-references inside this widget:** its blurbs cite "Section 4" for pre-training,
"Section 12" for label smoothing, "Section 15" for the RLHF penalty and "Section 19" for
distillation. In the current lesson those are Sections 5, 14, 17 and 21. The lesson's own §22 table
likewise cites "Sections 9 and 12" for z-loss where the text has it in 11 and 14. The widget copy
was written against an earlier section numbering; trust the lesson body, not the citations.

## widget_13_draft_verify.html — MTP as draft-and-verify

Same trunk, seed and 10-sentence corpus as widget_5, heads fitted the same way (600 epochs, lr 3).

| Slider | min | max | step | default |
| ------ | --- | --- | ---- | ------- |
| `k` tokens covered per verification pass | 2 | 4 | 1 | 4 |

Acceptance logic: heads 2…K propose from the current trunk state; the truth is head 1 run
autoregressively K times; the accepted run stops at the **first** mismatch (prefix acceptance, not
per-token). The widget tracks per-head hit counts and the overall acceptance.

Lesson quotes from this widget: head 2 lands **75%**, head 4 **27%**, overall acceptance **50.7%**.

---

## Anchor numbers worth carrying into an implementation

| Quantity | Value | Source |
| -------- | ----- | ------ |
| V | 131,072 | widget_1, widget_7, widget_8 |
| d_model | 4,096 | widget_1, widget_7 |
| dense head parameters | 536,870,912 (536.9M) | widget_1 |
| head vs Session 7 input side | 16.0× | widget_1 |
| logits vs hidden states | 32.0× (V/D) | widget_7 |
| untrained-model loss anchor | ln(131,072) = 11.7835 | lesson §7, verified |
| full-pass logits memory, 65,536 tokens bf16 | 16 GiB | widget_8 |
| chunked at 1,024 tokens | 256 MiB, 64× reduction | widget_8 |
| logits + gradient at B=1, T=262,144 | 128 GiB — exceeds any single card | widget_7 |
| z-loss λ | 1e-3 | widget_4 |
| soft-cap c | 30 (Gemma 2: 30 on final logits, 50 on attention) | widget_4, lesson §11 |
| DPO β | 0.10 | widget_6 |
| finite-difference step for the gradient check | 1e-4, central | widget_2 |
| PPO resident memory at 7B | 234.7 GiB across four models | widget_11 |
| DPO / GRPO resident | 117.3 / 130.4 GiB | widget_11 |
| four dense MTP heads | 2.1B parameters | lesson §13, verified |
| GRPO advantage normalisation | **population** std (÷G), not sample std | lesson §19 worked example, verified |

The GRPO row is the one an implementation gets wrong quietly: the lesson's advantages
(+1.526, −0.723, +0.241, −1.044) for rewards (0.9, 0.2, 0.5, 0.1) reproduce only with mean 0.425
and std 0.3112 computed over G, not G−1.
