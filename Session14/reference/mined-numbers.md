# Session 14 — numbers mined from the widget sources

Everything below is read out of the inline `<script>` of each widget under `widgets/`. The simulation
end states (widgets 3 and 6) were reproduced by running the widgets' own code under node with their fixed
seeds. They are the baselines an implementation gets measured against, and several of them appear
nowhere in the lesson prose.

The reference model is the Qwen3-30B-A3B shape: 48 layers, hidden 2,048, 128 experts of width 768, top-8,
vocabulary 151,936. Its constants recur across widgets 2, 7 and 8:

| Name | Value | Meaning |
|---|---|---|
| `L` | 48 | layers |
| `ATT` | 18,874,624 | attention parameters per layer (Q widens 2,048 → 4,096) |
| `EXP` | `3*2048*768` = 4,718,592 | one expert (gate, up, down) |
| router | `2048 * E` per layer | 262,144 at E = 128 |
| `NORM` | 4,096 | two RMSNorm vectors per layer |
| `ONCE` | 622,329,856 + 2,048 | input + output embeddings, plus the final norm |
| `params(nE, used)` | `L*(ATT + used*EXP + HID*nE + NORM) + ONCE` | total with `used = nE`, active with `used = k` |
| TOTAL | 30,532,122,624 | 30.53B |
| ACTIVE | 3,353,032,704 | 3.35B |
| DENSE = TOTAL − 48·128·EXP | 1,541,093,376 | 1.54B (attention, router, norms, embeddings) |
| `BPW` | 16 bytes/param | mixed-precision training state |
| `GIB` | 1,073,741,824 | |
| card caps | H100 74.5 GiB, B200 167.6 GiB | |
| links | NVLink 4 = 450, NVLink 5 = 900, network card = 50 GB/s | |
| peak 16-bit | H100 0.989, B200 2.25 PFLOPS | |
| activations | `8192 × 2048 × 34 × 48` bytes | 25.5 GiB per 8,192-token sequence |

## widget_1_router

| Name | Value |
|---|---|
| tokens | 3 (A, B, C), 8 logits each |
| logit axis | `LMIN = -1.5`, `LMAX = 2.5` |
| score function buttons | softmax / sigmoid — default `softmax` |
| k buttons | 1, 2, 4 — default `k = 2` |
| default token | A (`tok = 0`) |

Logit tables:

| token | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
|---|---|---|---|---|---|---|---|---|
| A | 2.0 | 0.5 | 1.2 | −0.3 | 0.9 | 1.8 | −1.0 | 0.1 |
| B | −0.4 | 0.8 | 0.2 | 2.3 | −0.6 | 0.3 | 1.1 | −0.2 |
| C | 0.3 | −0.5 | 1.1 | 0.6 | 0.2 | −0.8 | 0.9 | 1.6 |

Formulas: softmax `e^z_i / Σ e^z`; sigmoid `1 / (1 + e^−z_i)`; kept weight `score_i / Σ kept scores`,
non-kept shown as "dropped". Top-2 results (a router implementation should reproduce these exactly):

| token | softmax top-2 → weights | sigmoid top-2 → weights |
|---|---|---|
| A | e0, e5 (0.320, 0.262) → 0.550, 0.450 | e0, e5 (0.881, 0.858) → 0.507, 0.493 |
| B | e3, e6 (0.503, 0.152) → 0.769, 0.231 | e3, e6 (0.909, 0.750) → 0.548, 0.452 |
| C | e7, e2 (0.312, 0.189) → 0.622, 0.378 | e7, e2 (0.832, 0.750) → 0.526, 0.474 |

Sigmoid flattens the weights toward 0.5/0.5; softmax exaggerates the gap.

## widget_2_active_total

| Name | Value |
|---|---|
| experts buttons | 32, 64, 128, 256 — default `E = 128` |
| top-k buttons | 2, 4, 8, 16 — default `k = 8` |
| bar axis `AXMAX` | 64e9 params |
| lit-expert shuffle | `mulberry32(nE*1000 + n*7 + 14)` |
| card count | `ceil(state / 167.6)` B200 cards |

Readouts: total, active, `total/active`, `6 × active` GFLOP/token, `total × 16 / 2^30` GiB.

| E \ k | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| 32 (T 8.78B, 130.8 GiB, 1 card) | A 1.98B · 4.4× · 11.9 GF | 2.44B · 3.6× · 14.6 | 3.34B · 2.6× · 20.1 | 5.16B · 1.7× · 30.9 |
| 64 (T 16.03B, 238.9 GiB, 2 cards) | 1.99B · 8.1× · 11.9 | 2.44B · 6.6× · 14.6 | 3.35B · 4.8× · 20.1 | 5.16B · 3.1× · 31.0 |
| 128 (T 30.53B, 455.0 GiB, 3 cards) | 1.99B · 15.3× · 12.0 | 2.45B · 12.5× · 14.7 | **3.35B · 9.1× · 20.1** | 5.16B · 5.9× · 31.0 |
| 256 (T 59.54B, 887.2 GiB, 6 cards) | 2.01B · 29.7× · 12.0 | 2.46B · 24.2× · 14.8 | 3.37B · 17.7× · 20.2 | 5.18B · 11.5× · 31.1 |

Active barely moves with E (only the router grows); total barely moves with k.

## widget_3_collapse

Toy simulation, 16 experts, top-2, seeded.

| Name | Value | Meaning |
|---|---|---|
| `E`, `T`, `K` | 16, 256, 2 | experts, tokens per step, top-k |
| `MEAN` | `T*K/E` = 32 | even load per expert |
| `END` | 300 | steps per run |
| `WIN` | 20 | window for averaged load and dead count |
| `SEED` | 14 | |
| `SC` | 0.5 | std of each expert's fixed initial bias `c_j` |
| `SN` | 1 | std of per-token routing noise |
| `AL`, `DC` | 0.08, 0.01 | rich-get-richer: `s_j += 0.08·load_j/256 − 0.01·s_j` |
| `LAM` | 0.3 | aux-loss mode: `s_j -= 0.3·(share_j − 1/16)` |
| `GAM` | 0.02 | bias mode: `b_j += 0.02·sign(32 − load_j)` (widget notes real runs use 0.001) |
| modes | 0 no balancing (default), 1 auxiliary loss, 2 bias method | |
| dead | 0 tokens in each of the last 20 steps | |
| MaxVio colour gate | < 0.3 green, ≤ 1 amber, > 1 red | |

Choose rule: top-2 of `s_j + c_j + noise` (+ `b_j` in mode 2 only). End state at step 300:

| mode | dead experts | MaxVio | loads (last-20 average) |
|---|---|---|---|
| no balancing | 14 of 16 | 7.00 | two experts take all 256 each, rest 0 |
| auxiliary loss | 0 | 0.51 | 15 … 48 |
| bias method | 0 | 0.06 | 30 … 34 |

## widget_4_capacity

| Name | Value |
|---|---|
| `LOAD` | 118, 96, 88, 80, 74, 70, 66, 62, 58, 54, 50, 46, 44, 42, 40, 36 |
| `E`, `TOT`, `AVG` | 16, 1,024 assignments, 64 |
| capacity | `floor(1024 / 16 × cf)` |
| cf buttons | 1.0, 1.25, 1.5, 2.0, dropless (`cf = 0`) — default `cf = 1` |
| drop colour gate | 0 green, < 5% amber, ≥ 5% red |

| cf | capacity | dropped | share | experts over | slots reserved |
|---|---|---|---|---|---|
| 1.0 | 64 | 144 | 14.1% | 7 | 1,024 |
| 1.25 | 80 | 62 | 6.1% | 3 | 1,280 |
| 1.5 | 96 | 22 | 2.1% | 1 | 1,536 |
| 2.0 | 128 | 0 | 0.0% | 0 | 2,048 |
| dropless | ∞ | 0 | 0.0% | 0 | 1,024 |

## widget_5_scope

| Name | Value |
|---|---|
| `E`, `EVEN` | 16, 6.25% |
| subject mix | code 0.5, maths 0.25, prose 0.25 |
| modes | `micro` (default) / `global` |
| self-check | each row sums to 100, each expert's weighted load = 6.25 (throws otherwise) |

Micro mode: every cell 6.25%. Global mode (% of that subject's tokens per expert E1–E16):

| subject | E1–E4 | E5–E8 | E9–E16 |
|---|---|---|---|
| code | 1, 1.5, 2, 2.5 | 1, 1.5, 2, 2.5 | 11.5, 11.5, 11, 11, 10.5, 10.5, 10, 10 |
| maths | 22, 21, 20, 19 | 1, 1, 1, 1 | 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5 |
| prose | 1, 1, 1, 1 | 22, 21, 20, 19 | 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5 |

Global: max per subject code 11.5%, maths 22%, prose 22%; E1–E4 hold 82% of maths, E5–E8 82% of prose;
total load still exactly 6.25% everywhere (MaxVio 0 in both modes).

## widget_6_growth

Stage 1, dense → 4 experts (8 slices a–h):

| method | layout | readouts |
|---|---|---|
| copy (default) | all 8 slices each | 0 of 3 experts differ; "experts start identical" |
| partition | random half each, `mulberry(7)`: {b,c,e,g}, {a,c,f,g}, {a,d,e,f}, {c,d,e,g} + a shared expert with all 8 | miss chance `(1 − 4/8)^4` = 6.25%; this draw misses slice h; with 20 experts `(1/2)^20` = 1 in 1,048,576 |
| drop | copy with half redrawn, `mulberry(1007)`: redraw {d,e,f,h}, {b,c,d,f}, {a,b,d,h}, {a,b,c,f} | keeps 4, redraws 4, r = 0.5 |

Stage 2, 4 experts → 24 clones (4 families × 6):

| Name | Value | Meaning |
|---|---|---|
| `STEPS` | 200 | |
| `SOFT` | 100 | probabilistic window (steps 1–100), then hard top-k |
| `TOK` | 128 | tokens per step, top-2 → 256 slots |
| `WIN` | 20 | dead = 0 tokens over last 20 steps, counted from step 20 |
| `E`, `SUB` | 24, 12 | clones, token sub-types |
| family pull | +3 score to the token's family | |
| clone init | `gauss·0.02` | |
| per-token noise | `gauss·0.005` | |
| specialization EMA | `s = 0.98·s + 0.02·count` | |
| routing buttons | hard (default) / prob | seed `mulberry(7)` |

End state at step 200: **hard top-k → 14 of 24 dead (4, 3, 3, 4 per family); probabilistic → 0 dead.**
Note text: Lightning LM 20 → 460 by 23 clones each; clone-only + hard top-k reached 168 dead of 460.

## widget_7_all_to_all

| Name | Value |
|---|---|
| `K`, `H`, `BYTES`, `SEQ`, `L` | 8, 2048, 2, 8192, 48 |
| `G`, `E`, `PER` | 8 GPUs, 128 experts, 16 per GPU (map `expert / 16`) |
| `SEED` | 52 (made-up routing) |
| link buttons | nv5 (default, 900 GB/s, B200 2.25 PF), nv4 (450, H100 0.989 PF), net (50, B200 2.25 PF) |
| phases | route → dispatch → compute → combine |

Per token copy bytes `8 × 2048 × 2` = 32,768. Traffic `2 × 32768 × 8192 × 7/8 × 48 × 2` = 45.1 GB.
Compute `6 × ACTIVE × 8192` = 164.8 TFLOP. Ratio colour: > 1 red.

| link | comm | compute | comm / compute |
|---|---|---|---|
| NVLink 5 | 0.050 s | 0.073 s | 0.68 |
| NVLink 4 | 0.100 s | 0.167 s | 0.60 |
| network card | 0.902 s | 0.073 s | 12.31 |

## widget_8_layout

| Name | Value |
|---|---|
| node buttons | h100, b200 — default `b200` |
| EP buttons | 8 (1 node), 16 (2 nodes), 32 (4 nodes) — default `ep = 8` |
| experts per GPU | `EXP_total / ep × 16 / GiB` |
| dense per GPU | `DENSE × (4 + 12/gpus) / GiB` (ZeRO-1 over all GPUs) |
| traffic | `51.5 GB × share`; share 7/8 at EP 8 (NVLink), `(ep−8)/ep` across nodes (network, 50 GB/s) |
| tokens per expert | `gpus × 8192 × 8 / 128` |
| axes | memory 180 GiB, time 1.0 s |

| node, EP | experts | dense | state | +act 25.5 | fits | traffic | comm | compute | ratio | tok/expert |
|---|---|---|---|---|---|---|---|---|---|---|
| B200, 8 | 54.0 | 7.9 | 61.9 | 87.4 | yes (80.2 free) | 45.1 GB | 0.050 | 0.073 | 0.68 | 4,096 |
| B200, 16 | 27.0 | 6.8 | 33.8 | 59.3 | yes | 25.8 GB | 0.515 | 0.073 | 7.04 | 8,192 |
| B200, 32 | 13.5 | 6.3 | 19.8 | 45.3 | yes | 38.7 GB | 0.773 | 0.073 | 10.55 | 16,384 |
| H100, 8 | 54.0 | 7.9 | 61.9 | 87.4 | **no** | 45.1 GB | 0.100 | 0.167 | 0.60 | 4,096 |
| H100, 16 | 27.0 | 6.8 | 33.8 | 59.3 | yes | 25.8 GB | 0.515 | 0.167 | 3.09 | 8,192 |
| H100, 32 | 13.5 | 6.3 | 19.8 | 45.3 | yes | 38.7 GB | 0.773 | 0.167 | 4.64 | 16,384 |

Only B200 EP 8 keeps comm below compute on a card that fits; every multi-node layout is comm-bound.
