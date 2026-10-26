# Session 15 — numbers mined from the widget sources

Everything below is read out of the inline `<script>` of each widget under `widgets/`. The end states
were reproduced by running each widget's own code under node against a stub DOM (fixed seeds, so they are
deterministic). The widgets are buttons only — no sliders — so the "defaults" are the button that is lit
on load.

The reference model is the 2026 open MoE shape of the lesson: 50 layers, hidden 2,560, 384 routed + 1
shared expert of width 512, top-6, vocabulary 128,000, 78.1B total / 3.46B active, trained on 768 B200 at
75.5M tokens and 6 s per step. Constants that recur:

| Name | Value | Where |
|---|---|---|
| `STEP` | 6 s per step | w3, w5, w8 |
| `SAVE` | 1 min per checkpoint save | w3, w5 |
| `RESTART` | 37.0 min fixed cost per failure | w3, w5 |
| `MTBF` | 12.9 h (our run) | w3 |
| failure rate, ours | 38 / 377,000 GPU-h = 1.01 per 10,000 GPU-h | w5 |
| failure rate, Llama 3 | 419 / (54 × 24 × 16,384) = 419 / 21,233,664 = 0.20 per 10,000 GPU-h | w5 |
| bytes per weight | 16 trained, 2 frozen | w4 |
| `PEAK` | 2.25e15 FLOP/s (B200 BF16) | w8 |
| `TOK` | 75.5e6 tokens per step | w8 |
| `NACT` | 3.46e9 active params | w8 |
| PRNG | mulberry32 everywhere | all |

## widget_1_logits — attention logits under four fixes

Buttons: `none` (default) / `norm` QK-norm / `clip` QK-Clip τ = 100 / `cap` soft-cap 50. 10,000 steps,
sampled every 10. Seed 1515.

| Curve | Formula |
|---|---|
| raw max logit | `10^(1.3 + 2.6·u^0.88 + 0.025·g)`, `u = t/10000` |
| QK-norm | `clamp(22 + 9(1 − e^(−t/900)) + 4 sin(t/1300) + 1.2g, 20, 40)` |
| QK-Clip | `20·e^(t/750)` until it hits 100 (t < 3,000), then `min(99.5, 100·e^(−(t−3000)/9000))` |
| soft-cap | `50·tanh(raw/50)` |
| loss trend | `2.25 + 5.6e^(−t/550) + 0.5e^(−t/4000)` |
| spike rule | a spike is scheduled 200–500 steps after raw logit first passes 1,000, then every 500–950 steps; height `0.9 + 1.6r`, decay τ = 60 steps; counted when loss − trend ≥ `SPK = 0.5` |

| Mode | final logit | loss spikes | first step logit > 1,000 |
|---|---|---|---|
| none | 7,484 | 6 | 5,990 |
| QK-norm | 36.3 | 0 | never |
| QK-Clip | 44.2 | 0 | never |
| soft-cap | 50.0 (= 50·tanh(7,484/50)) | 0 | never (pre-cap logit still grows) |

Only the no-fix run spikes; the others use the trend loss with no spikes added.

## widget_2_detector — three spike rules on one grad-norm trace

3,000 steps, seed 2. Trace: `b = 0.62 − 0.2t/3000 + 3.7e^(−t/85)`, multiplicative noise `±4.5%`
(uniform-sum). Warmup band shaded to step 300.

Injected spikes `[step, +height, label]`:
`[700, 0.16, small] [1100, 2.6, large] [1500, 0.55, medium] [1950, 0.14, small] [2350, 0.6, medium] [2750, 3.4, large]`

| Rule | Parameters | found | false alarms |
|---|---|---|---|
| fixed limit (default) | `LIM = 3.0` | 2 of 6 (both large) | 39, all in warmup |
| 7-sigma rolling | `KS = 7`, window `WIN = 1,000`, min `MINW = 100`, `|z| ≥ 7` | 4 of 6 (misses both small) | 0 |
| EMA z-score | `ALPHA = 0.97`, `ZE = 2.5`, one-sided `z > 2.5` | 6 of 6 | 2 (steps 2 and 1,591) |

Both adaptive rules exclude an alarmed value from the running mean/std. EMA update:
`d = x − m; m += (1−α)d; v = α(v + (1−α)d²)`.

## widget_3_rollback — checkpoint interval and the cost per day

Spike at step `SPIKE = 2,410`. Interval buttons 100 / **250** / 500 / 1,000; rollback buttons **last
checkpoint before the spike** / one checkpoint earlier.

Formulas:

- saves per day = `86,400 / (6 × interval)`; save minutes = that × 1 min
- failures per day `FPD = 24 / 12.9 = 1.86`
- failure minutes = `FPD × (37.0 + interval/2 × 6/60)` — restart plus half an interval of lost work
- total = save + failure minutes
- Young's interval: `sqrt(2 × 1 × 12.9 × 60) × 60 / 6` = **about 390 steps**

| Interval | steps lost (last ckpt) | saves/day | save min | failure min | total min/day | % of day |
|---|---|---|---|---|---|---|
| 100 | 10 | 144.0 | 144.0 | 78.1 | 222 | 15.4% |
| 250 | 160 (16.0 min) | 57.6 | 57.6 | 92.1 | 150 | 10.4% |
| 500 | 410 | 28.8 | 28.8 | 115.3 | 144 | 10.0% |
| 1,000 | 410 | 14.4 | 14.4 | 161.9 | 176 | 12.2% |

Rolling back one checkpoint earlier at 1,000 loses 1,410 steps = 2.35 h. The cost curve is plotted over
50–1,100 steps; its minimum sits near Young's 390.

## widget_4_frozen — memory of a partly frozen model

Parameter counts per part (sum 78.10e9):

| Part | Params | Toggle |
|---|---|---|
| routed experts | 75,497,472,000 | yes |
| shared experts | 196,608,000 | no |
| attention | 1,703,936,000 | yes |
| routers | 49,152,000 | yes |
| embeddings and LM head | 655,360,000 | yes |
| norms and biases | 546,560 | no |

Memory = trainable × 16 B + frozen × 2 B, over 2^30. Zero-gradient count per part is a fixed random
`3 + floor(r·58)` (seed 1515) when trainable, "no gradient" when frozen.

| Frozen set (cumulative) | trainable GiB | frozen GiB | total GiB | constraints listed |
|---|---|---|---|---|
| nothing (default) | 1,163.8 | 0 | 1,163.8 | 0 |
| routed | 38.8 | 140.6 | 179.5 | 4 |
| + attention | 13.4 | 143.8 | 157.2 | 5 |
| + routers | 12.7 | 143.9 | 156.6 | 6 |
| + embeddings | 2.9 | 145.1 | 148.0 | 7 |

Constraint list: any freeze → optimizer from trainable only, re-check grad-norm limits, hash frozen
weights; embeddings → non-reentrant activation checkpointing; routers → MiMo-V2.6 RL; routed → merge
adapters with LR re-warm; attention → FSDP1 separate wrapped units.

## widget_5_failures — failure rate against cluster size

GPU buttons 64 / 256 / **768** / 2,048 / 16,384; interval 100 / **250** / 500 / 1,000; source **ours** /
Llama 3. Colour gate on share of day lost: green < 15%, amber 15–100%, red ≥ 100%.

`failures/day = GPUs × 24 × rate`; `lost min/day = failures/day × (37.0 + iv/2 × 6/60) + 86,400/(6·iv) × 1`

At interval 250:

| GPUs | ours: failures/day | MTBF | lost min/day | % | Llama 3 rate: lost min/day | % |
|---|---|---|---|---|---|---|
| 64 | 0.15 | 155.0 h | 65 | 4.5% | 59 | 4.1% |
| 256 | 0.62 | 38.8 h | 88 | 6.1% | 64 | 4.4% |
| 768 | 1.86 | 12.9 h | 150 | 10.4% | 76 | 5.3% |
| 2,048 | 4.95 | 4.8 h | 303 | 21.0% | 106 | 7.3% |
| 16,384 | 39.63 | 0.61 h | 2,020 | 140.2% (red: cannot progress) | 442 (7.76/day, 3.1 h) | 30.7% |

## widget_6_misload — first-step loss of a silent misload

| Name | Value |
|---|---|
| `L1G` correct load, step 1 | 2.55 nats (measured) |
| `L1B` bad load, step 1 | 2.95 nats (measured) |
| gap | 0.40 nats |
| noise `SD` | 0.05 (z normalised to exactly std 1) |
| gap / noise | **8.0 σ** |
| trend | `2.55 − 0.028·ln(step)` |
| bad curve | trend + 0.05 + 0.35·e^(−(step−1)/30) + noise — closes to a 0.05 offset |
| steps | 200, seed 2026; band = trend ± 2σ |

Steps 2–200 are labelled as simulation; only step 1 values are measured.

## widget_7_downsample — keeping a spike in 2,000 pixels

300,000 steps into 1,000 columns of `B = 300` steps. Seed 15007. Curve `2.0 + 2.0e^(−i/30000)` + 0.04
gaussian noise. Spikes: single step `S1 = 187,431` to `V1 = 8.0`; five steps from `S5 = 62,000` to
`V5 = 4.5`. "Visible" = drawn column max ≥ midpoint of base and peak (5.00 and 3.38).

| Method | points drawn | col max at S1 | 8.0 visible | 4.5 visible |
|---|---|---|---|---|
| mean (default) | 1,000 | 2.023 | no | no (2.29) |
| every 300th | 1,000 | 2.079 | no | no |
| min and max | 2,000 | 8.000 | yes | yes |
| M4 (first, min, max, last) | 3,986 | 8.000 | yes | yes |

Fact lines: `(999 × 2.0 + 8.0)/1,000 = 2.006`; rollups at 10/100/1,000/10,000 add
30,000 + 3,000 + 300 + 30 = 33,330 rows = **11.1%** of 300,000.

## widget_8_dashboard — replay with scripted incidents

3,000 steps on 768 GPUs, checkpoint every `CKPT = 100`, restart gap `RESTART_S = 900 s`, seed 150815.
Records: steps 1–2,200, restart, steps 2,101–3,000 on 760 GPUs (3,100 records). MFU =
`6 × 3.46e9 × tok/s / (GPUs × 2.25e15)`. Healthy tokens/s = 75.5M / 6 s = 12.58M.

Traces: loss `2.6 + 1.4e^(−s/900)`; grad norm `(0.35 + 0.65e^(−s/600))`; attention logit `30 + 10s/3000`
(never near 100); MaxVio per layer L0/L12/L25/L49 `(0.25 + 0.08l) + 0.45e^(−s/700)`; GPU temp 72 °C.

Scripted incidents and the alert each fires:

| Step | Incident (script) | Severity | Alert fired |
|---|---|---|---|
| 700 | grad norm × 1.35 (`SPIKE`) | warn | EMA z-score 11.8 > 2.5 |
| 1,200 | layer 0 intervention rate jumps 0.18 → 0.905+ (`ROUTE`), MaxVio stays 0.18 | warn | rate 90.8% > 90%, MaxVio 0.17 — balance looks fine |
| 1,700–1,900 | straggler: step time 6/0.88 s, hottest GPU 87 °C (`STRAG0/1`) | warn | tokens/s drop 12% > 10% for 100 steps (fires at 1,800) |
| 2,200 | Xid 79 on node 41 (`XID_STEP`) | page | remove node 41, restart from 2,100 |
| 2,600 | frozen router weights change (`HASH_STEP`) | stop | hash differs from step 0 |

Spike score (7σ over last 1,000) never fires on this replay. tokens/s rule: last 100 records all below
0.9 × mean of records i−600 … i−200. Zero-gradient panel tracks attention 200, routed experts 57,600,
shared expert 150 tensors (all 0 in the replay).
