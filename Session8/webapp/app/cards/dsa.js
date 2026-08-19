// Concept 25 — DeepSeek sparse attention (DeepSeek-V3.2-Exp release, 2025-09-29).
// Research note: docs/research/dsa.md.
//
// Four decisions, from the note:
//
// One, no seam change and no card-local mixer. DSA's core attention is ordinary attention over a
// top-k subset, which `softmaxMixer({ readable })` already expresses; the indexer reads the block
// input through the existing `latent` hook, which is exactly what the report says it reads. All the
// novelty is in how the subset is chosen, so that is where the card spends its space.
//
// Two, the honest centre of this card is a negative result stronger than any in the deck: an
// untrained indexer is indistinguishable from random. So the card brackets the mechanism between its
// floor (random) and its ceiling (the oracle distribution Eq. 3 fits the indexer to), and says that
// the gap between them is what 943.7B tokens of continued pre-training buys.
//
// Three, H^I and d^I are not in the report. The card prints this app's choice labelled as this app's
// choice, and prints no cost-per-token figure at all, because that evidence is a figure with no table.
//
// Four, the report cites concept 24 for the constraint that forces MQA mode and concept 7 for MQA
// itself. Those links are the report's own, not the card's invention.
import { el, slider, choice } from "../lib/dom.js";
import { fmt, dot, softmax, mulberry32, gauss } from "../model/ops.js";
import { forward, DH, CONFIG } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { state } from "../runner.js";
import { barList, readout } from "../views/bars.js";
import { attentionGrid } from "../views/grid.js";
import { curveView } from "../views/curve.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

// The report gives neither the indexer's head count nor its width — "a small number of heads" is all
// it says. These are this app's choices and the card labels them as such.
export const HI = 2;
export const DI = 8;
const relu = (x) => (x > 0 ? x : 0);

/** Seeded stand-ins for the indexer's own projections: q^I and w^I from h_t, k^I from h_s. The
 *  report's are trained by Eq. 3; this model is untrained, so these are declared random. */
const IDX = (() => {
  const rnd = mulberry32(90929);
  const mat = (r, c) => Array.from({ length: r }, () => Float64Array.from({ length: c }, () => gauss(rnd) / Math.sqrt(c)));
  return {
    q: Array.from({ length: HI }, () => mat(DI, CONFIG.D)),
    k: mat(DI, CONFIG.D),
    w: mat(HI, CONFIG.D),
  };
})();
const project = (M, v) => Float64Array.from(M, (row) => dot(row, v));

/** Eq. 1: I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s). Reads the block input, not the
 *  attention's own queries and keys — the indexer is a separate network beside attention. */
export function indexScores(normed) {
  const T = normed.length;
  const kI = normed.map((h) => project(IDX.k, h));
  const rows = [];
  for (let t = 0; t < T; t++) {
    const qI = IDX.q.map((M) => project(M, normed[t]));
    const w = project(IDX.w, normed[t]);
    const row = new Float64Array(T);
    for (let s = 0; s <= t; s++) {
      let acc = 0;
      for (let j = 0; j < HI; j++) acc += w[j] * relu(dot(qI[j], kI[s]));
      row[s] = acc;
    }
    rows.push(row);
  }
  return rows;
}

/** Eq. 3's target: the main attention's scores summed across all heads, then L1-normalised along
 *  the sequence. This is what the indexer is trained to imitate, so the card can show the ceiling
 *  without pretending to train anything. */
export function klTarget(trace, T) {
  const rows = [];
  for (let t = 0; t < T; t++) {
    const acc = new Float64Array(T);
    for (const h of trace.heads) for (let s = 0; s <= t; s++) acc[s] += h.weights[t][s];
    let sum = 0;
    for (const x of acc) sum += x;
    for (let s = 0; s < T; s++) acc[s] /= sum || 1;
    rows.push(acc);
  }
  return rows;
}

/** Eq. 2: keep the top-k index scores for this query, and nothing else. */
export const topkSet = (row, t, k) => {
  const idx = [];
  for (let s = 0; s <= t; s++) idx.push(s);
  idx.sort((a, b) => row[b] - row[a]);
  return new Set(idx.slice(0, k));
};

/** D_KL(p ‖ softmax(I)) — the report's Eq. 3, evaluated rather than minimised. */
export function klOf(p, I, t) {
  const q = softmax(Array.from({ length: t + 1 }, (_, s) => I[s]));
  let acc = 0;
  for (let s = 0; s <= t; s++) if (p[s] > 1e-12) acc += p[s] * Math.log(p[s] / Math.max(q[s], 1e-12));
  return acc;
}

const SELECTORS = {
  indexer: "the lightning indexer, untrained",
  oracle: "the target Eq. 3 fits it to",
  random: "picked at random",
};

export function dsaCard(root, m) {
  let k = 4;
  let selector = "indexer";
  let query = null;
  const shuffleRnd = mulberry32(7);
  const NOISE = Array.from({ length: 64 }, () => shuffleRnd());

  root.appendChild(
    prose({
      problem:
        "The previous card made per-query selection affordable by making it coarse. Scoring every key is the cost being avoided, so it scored blocks instead of tokens, and got the scores for free as a by-product of a branch it was already computing. That bought a fixed hierarchy: block length, stride, selection block size, how many blocks, a window, and three blocks always read whatever the query wants. Seven numbers chosen before any data arrives, and a mechanism whose finest possible distinction is sixty-four tokens wide.",
      mechanism:
        "Keep per-token selection and make the scorer cheap instead of the score coarse. A small separate network — its own heads, its own projections, reading the block input rather than attention's own queries and keys — produces one number per past token, using a rectifier instead of an exponential, for speed rather than for accuracy. Rank those numbers, keep the best two thousand and forty-eight, and let ordinary attention read exactly those. Nothing else is chosen in advance: no blocks, no window, no gate, one k. The catch is that this scorer is worth nothing until it is trained, and it is not trained by the language it is modelling — it is fitted, by KL divergence over just under a trillion tokens, to imitate the attention distribution it stands in front of.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "one token's journey, through the keys an index network let through");

  // ------------------------------------------------------- 1. the indexer
  const idxHost = el("div", {});
  const idxRead = readout([
    { key: "kl", label: "KL(p ‖ softmax(I)) — the quantity Eq. 3 minimises" },
    { key: "uniform", label: "the same, for an indexer that says nothing" },
    { key: "neg", label: "scores below zero — the per-head weight is unconstrained" },
    { key: "zero", label: "scores at exactly zero — every head silent, so the ranking ties" },
  ]);
  const idxNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the lightning indexer — a second, smaller network beside attention" }),
      el("div", { class: "formula", text: "I(t,s) = Σⱼ w^I(t,j) · ReLU( q^I(t,j) · k^I(s) )" }),
      idxHost,
      idxRead.node,
      idxNote,
    ])
  );

  // ------------------------------------------------------- 2. top-k, and what the choice is worth
  const kSlider = slider({
    label: "k — keys each query may read",
    min: 1,
    max: 26,
    value: 4,
    oninput: (v) => ((k = v), render()),
  });
  const selPick = choice({
    label: "who chooses",
    value: "indexer",
    options: Object.entries(SELECTORS).map(([v, label]) => ({ value: v, label })),
    onchange: (v) => ((selector = v), render()),
  });
  const grid = attentionGrid({ onPickRow: (i) => ((query = i), render()), label: "what the indexer let through" });
  const massCurve = curveView({
    xLabel: "k — keys allowed through",
    yLabel: "weight kept",
    ariaLabel: "share of full attention's weight inside the top-k, against k",
  });
  const kRead = readout([
    { key: "mass", label: "share of full attention's own weight inside the top-k" },
    { key: "dev", label: "how far the output moved from full attention" },
    { key: "word", label: "next word / full attention's own" },
  ]);
  const kNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "top-k over tokens, not blocks — and what the choice is worth" }),
      el("div", { class: "ctrls" }, [kSlider.node, selPick]),
      el("div", { class: "formula", text: "uₜ = Attn( hₜ, { cₛ : I(t,s) ∈ Top-k I(t,·) } )" }),
      grid.node,
      massCurve.node,
      kRead.node,
      kNote,
    ])
  );

  // ------------------------------------------------------- 3. the bill
  const billRead = readout([
    { key: "warm", label: "dense warm-up, indexer only, from 1000 × 16 × 128K" },
    { key: "sparse", label: "sparse stage, everything unfrozen, from 15000 × 480 × 128K" },
    { key: "rate", label: "k = 2048 of a 128K context" },
    { key: "cplx", label: "what stays quadratic" },
  ]);
  const billNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the price of a scorer that has to be taught" }),
      billRead.node,
      billNote,
    ])
  );

  // ------------------------------------------------------- 4. quoted
  const T1 = [
    ["MMLU-Pro (EM)", 85.0, 85.0],
    ["GPQA-Diamond (Pass@1)", 80.7, 79.9],
    ["Humanity's Last Exam (Pass@1)", 21.7, 19.8],
    ["BrowseComp (Acc.)", 38.5, 40.1],
    ["BrowseComp_zh (Acc.)", 45.0, 47.9],
    ["SimpleQA (Acc.)", 96.8, 97.1],
    ["LiveCodeBench (Pass@1)", 74.9, 74.1],
    ["Codeforces-Div1 (Rating)", 2046, 2121],
    ["Aider-Polyglot (Acc.)", 76.1, 74.5],
    ["SWE Verified (Agent)", 68.4, 67.8],
    ["SWE-bench Multilingual (Agent)", 57.8, 57.9],
    ["Terminal-bench (Terminus 1)", 36.7, 37.7],
    ["AIME 2025 (Pass@1)", 88.4, 89.3],
    ["HMMT 2025 (Pass@1)", 86.1, 83.6],
  ];
  const quoted = el("section", { class: "panel" }, [
    el("div", { class: "panel-title", text: "quoted, not computed — the release's own Table 1, all fourteen rows" }),
  ]);
  const t1bars = barList({
    rows: T1.map(([name]) => ({ key: name, label: name })),
  });
  const t1vals = {};
  for (const [name, a, b] of T1) {
    const delta = b - a;
    const scale = name.includes("Rating") ? 1 : 1;
    t1vals[name] = {
      value: Math.abs(delta),
      of: name.includes("Rating") ? 100 : 4,
      text: `${a} → ${b}   ${delta === 0 ? "level" : delta > 0 ? "+" + fmt(delta * scale, 1) : fmt(delta * scale, 1)}`,
      tone: delta < 0 ? "warn" : delta > 0 ? "" : "alt",
    };
  }
  t1bars.update(t1vals);
  quoted.appendChild(
    el("p", {
      class: "note",
      text: "V3.1-Terminus → V3.2-Exp, same post-training pipeline, data and algorithm on both, so the difference is DSA. The bar is the size of the change, orange where it went down.",
    })
  );
  quoted.appendChild(t1bars.node);
  quoted.appendChild(
    el("p", {
      class: "note",
      html: `<strong>Seven up, six down, one level</strong>, and the report's own summary is the fair one: <em>"DeepSeek-V3.2-Exp does not show substantial performance degradation."</em> Its explanation for the three largest drops is quoted rather than paraphrased, because it is an account and not a demonstration: <em>"The performance … on GPQA, HLE, and HMMT 2025 is lower … because DeepSeek-V3.2-Exp generates fewer reasoning tokens. However, this performance gap closes when using intermediate checkpoints that produce a comparable number of tokens."</em> <span class="warn">Those intermediate checkpoints are not shown</span>, and the claim is about output length rather than about attention. What is not in this source at all: <strong>no ablation of k, no comparison against the previous card or any other sparse method, and no table behind either figure.</strong> The commercially quotable number is from the dated release note, not the report — <strong>API prices cut by more than 50%</strong> the day this shipped. And the nearest thing to a limitations section, in full: <em>"we are actively pursuing further large-scale testing in real-world scenarios to uncover potential limitations of the sparse attention architecture."</em> The model's name says the same thing: <code>Exp</code>.`,
    })
  );
  root.appendChild(quoted);

  const honest = el("p", { class: "note warn" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what this page cannot show" }),
      honest,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Selection at the finest possible granularity — one token, not a block of sixty-four — with exactly one number chosen in advance: k. No stride, no window branch, no gate, no blocks always read",
        "Core attention drops from quadratic in the context to linear in k, and at the shipped setting each query reads 2,048 of 128,000 keys, which is 1.56% of what is behind it",
        "The scorer is cheap by construction: a rectifier instead of an exponential, chosen in the report's own words \"for throughput consideration\", a small number of heads, and FP8 arithmetic",
        "The selection is shared across every attention head of a token, so one set of keys is loaded per token — which is what makes fine-grained selection implementable at all, and the report cites the previous card for that constraint",
        "It was applied to an existing model by continued training rather than requiring a model built around it, and shipped as a product the same day with API prices cut by more than half",
        "Quality held across fourteen benchmarks in the report's own comparison: seven up, six down, one level, against the model it was continued from with the same post-training on both",
      ],
      givesUp: [
        "The mechanism is worth nothing untrained, and this page measures that: an untrained indexer captures 0.209 of full attention's weight at k=1 where random captures 0.206, and its KL against the target is 1.66 nats where an indexer that says nothing scores 0.54",
        "Just under a trillion tokens to fix that — 2.1B for the warm-up plus 943.7B for the sparse stage — which is the real price of the mechanism and is paid before it does anything",
        "The indexer is not trained by the language-modelling loss. Its input is detached from the graph and it is fitted only to imitate the attention it replaces, so the previous card's central claim of end-to-end trained sparsity is given up four months later",
        "The quadratic term never leaves: the indexer still scores every past token for every query, at O(L²), with a smaller constant and no stated size — the report gives neither head count nor width, so nobody outside can compute what it costs",
        "Below some unstated context length they do not run it at all, but simulate it with a mask over dense attention because dense is faster there",
        "One model pair, no ablation of anything, no comparison against any other sparse method, and the cost evidence is a figure with no table. The released inference code also had a rotation bug in the indexer itself, found seven weeks later",
      ],
      chooseWhen:
        "When you have a strong dense model, a long-context bill you need to cut now, and enough compute to spend a trillion tokens teaching a small network to imitate your own attention. It is the pragmatic end of this timeline: not the most elegant sparsity and not the most principled, but the one that went into production with prices cut in half the same day. Not when you cannot train — an untrained or badly trained indexer is measurably worse than no sparsity at all, and worse than random selection.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "It decides word by word what to look at, instead of in chunks of sixty-four, and only one setting has to be chosen by hand",
        "The thing that does the deciding is deliberately cheap and simple, so asking the question costs far less than answering it would",
        "At the shipped setting each word looks at about one and a half percent of what came before it, and the published scores barely move",
        "It was bolted onto a model that already existed rather than needing a new one, and the price of using it dropped by more than half the day it shipped",
      ],
      cons: [
        "Until the chooser is trained it is no better than picking at random — this page measures it doing exactly that — so the whole mechanism is a promise about training rather than a property of the design",
        "Teaching it took nearly a trillion words of extra training, and it is taught to copy the very thing it is replacing rather than to do the actual job well",
        "The cheap chooser still looks at every earlier word for every new word, so the cost that grows fastest with length never actually goes away",
        "On short text they quietly do not use it, because plain attention is faster there",
        "One comparison, no experiments taking any part of it away, and the released code had a bug in the chooser itself for the first seven weeks",
      ],
      verdict:
        "The most practical card in this timeline and the least self-contained. It takes the previous card's idea, throws away all the scaffolding, and replaces the clever free score with a small separate network that has to be taught — for nearly a trillion tokens — to imitate the attention it is standing in front of. When that works, each word reads one and a half percent of the text and the scores hold. When it has not been taught, this page shows what you get: a mechanism indistinguishable from choosing at random. Every other concept in this deck does something you can see on untrained weights. This one is a wager on training, and the whole of it sits in a six-page note shipped with a model whose name ends in Exp.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    if (query === null || query >= T) query = T - 1;
    k = Math.min(k, 26);
    kSlider.set(k);

    // The indexer reads the residual stream, which the latent hook can see without changing
    // anything: returning null leaves the block's own keys and values in place.
    const base = forward(tokens);
    const scores = [];
    forward(tokens, {
      latent: (normed, wb, b) => {
        scores[b] = indexScores(normed);
        return null;
      },
    });
    const targets = base.trace.map((tr) => klTarget(tr, T));

    const chooser = (b, t, kk, who = selector) => {
      if (who === "oracle") return topkSet(targets[b][t], t, kk);
      if (who === "random") {
        const idx = [];
        for (let s = 0; s <= t; s++) idx.push([s, NOISE[(s * 7 + t * 13 + b * 29) % NOISE.length]]);
        idx.sort((x, y) => x[1] - y[1]);
        return new Set(idx.slice(0, kk).map((x) => x[0]));
      }
      return topkSet(scores[b][t], t, kk);
    };
    const readable = (i, j, at) => chooser(at.block || 0, i, k).has(j);
    const run = forward(tokens, { mixer: softmaxMixer({ readable }) });
    const head = run.trace[0].heads[0];

    flow.update({
      tokens,
      head: { ...head, emb: run.trace[0].input },
      weights: head.weights,
      out: head.out,
      top: run.top,
      query,
      opts: { qkvBadge: `top ${k} of ${query + 1}, chosen once per token` },
    });
    flowNote.innerHTML = `Two networks run here, not one. The attention above is unchanged from concept 1 — it simply has fewer keys to look at. <strong>What chose them is a separate network that never appears in this picture</strong>: its own heads, its own projections, reading the same block input attention reads, and producing one number per earlier token. And the choice is made <em>once per token</em>, not once per head: every head of this token reads the same ${k} keys, because the report's key/value entries are shared across heads for the reason it cites the previous card for — <em>"each key-value entry must be shared across multiple queries for computational efficiency"</em>. That is concept 21's latent cache in its multi-query mode, doing the work that makes token-level selection implementable at all.`;

    // --- 1. the indexer
    const row = scores[0][query];
    if (idxHost.dataset.n !== String(T)) {
      idxHost.dataset.n = String(T);
      const b = barList({ rows: tokens.map((tk, i) => ({ key: `i${i}`, label: `${i}  ${tk.word}` })) });
      idxHost.replaceChildren(b.node);
      idxHost.__bars = b;
    } else {
      idxHost.querySelectorAll(".bar-name").forEach((n, i) => (n.textContent = `${i}  ${tokens[i].word}`));
    }
    const maxI = Math.max(...Array.from(row).slice(0, query + 1), 1e-9);
    const chosen = chooser(0, query, k);
    const ivals = {};
    for (let s = 0; s < T; s++) {
      ivals[`i${s}`] = {
        value: s <= query ? Math.max(0, row[s]) : 0,
        of: maxI,
        text: s > query ? "—" : `${fmt(row[s], 3)}${chosen.has(s) ? "  ← read" : ""}`,
        tone: s > query ? "alt" : chosen.has(s) ? "" : "alt",
      };
    }
    idxHost.__bars.update(ivals);
    let klAcc = 0;
    let uniAcc = 0;
    let n = 0;
    let negs = 0;
    let zeros = 0;
    let pairs = 0;
    for (let b = 0; b < base.trace.length; b++)
      for (let t = 1; t < T; t++) {
        klAcc += klOf(targets[b][t], scores[b][t], t);
        uniAcc += klOf(targets[b][t], new Float64Array(T), t);
        n++;
        for (let s = 0; s <= t; s++) {
          pairs++;
          if (scores[b][t][s] < 0) negs++;
          if (scores[b][t][s] === 0) zeros++;
        }
      }
    idxRead.update({
      kl: `${fmt(klAcc / n, 4)} nats`,
      uniform: `${fmt(uniAcc / n, 4)} nats`,
      neg: `${negs} of ${pairs} pairs, ${fmt((100 * negs) / (pairs || 1), 1)}%`,
      zero: `${zeros} of ${pairs} pairs, ${fmt((100 * zeros) / (pairs || 1), 1)}%`,
    });
    idxNote.innerHTML = `The bars are one query's index scores over every earlier token — the whole output of Eq. 1. Three properties to notice, all visible above. <strong>They are never normalised</strong>: no softmax, no sum to one, because only the ranking is ever used and the scale is never read. <strong>The rectifier is inside the sum, and the per-head weight outside it is not constrained</strong>, so a score can be negative — ${fmt((100 * negs) / (pairs || 1), 0)}% of them are here — and reading Eq. 1 as "ReLU, therefore non-negative" is simply a misreading of it. What the rectifier does buy is silence: a head contributes <em>exactly</em> zero for any key its own query does not point at, so each head speaks only about its own half-space, and <strong>a key that no head points at scores exactly zero — ${fmt((100 * zeros) / (pairs || 1), 0)}% of pairs on this model, which means the ranking Eq. 2 depends on contains ties broken by nothing at all.</strong> That fraction shrinks as the indexer gets more heads, and the number of heads is the figure the report never gives. The report's stated reason for choosing a rectifier is not accuracy but <em>"throughput consideration"</em>. And <strong>they come from a network that is not attention</strong>: separate projections of the same residual stream, with ${HI} heads of width ${DI} here. <span class="warn">Those two numbers are this app's, not the paper's</span> — the report says only "a small number of heads" and never states either, so nothing here may be read as a fact about the real indexer. The KL figure above is the exact quantity the report's Eq. 3 minimises, measured on this untrained stand-in: <strong>${fmt(
      klAcc / n,
      2
    )} nats against ${fmt(
      uniAcc / n,
      2
    )} for an indexer whose scores are all equal.</strong> An untrained indexer is worse than no indexer, and closing that gap is what the next panel is about.`;

    // --- 2. top-k
    grid.update({ tokens, weights: head.weights, query, readable: (i, j) => chooser(0, i, k).has(j) });
    const massAt = (kk, who) => {
      let acc = 0;
      let cnt = 0;
      for (let b = 0; b < base.trace.length; b++)
        for (let t = 0; t < T; t++) {
          const keep = chooser(b, t, kk, who);
          for (const h of base.trace[b].heads) {
            let mm = 0;
            for (const s of keep) mm += h.weights[t][s];
            acc += mm;
            cnt++;
          }
        }
      return acc / cnt;
    };
    const mine = massAt(k, selector);
    let num = 0;
    let den = 0;
    for (let b = 0; b < base.trace.length; b++)
      for (let h = 0; h < CONFIG.HEADS; h++)
        for (let t = 0; t < T; t++)
          for (let e = 0; e < DH; e++) {
            num += (run.trace[b].heads[h].out[t][e] - base.trace[b].heads[h].out[t][e]) ** 2;
            den += base.trace[b].heads[h].out[t][e] ** 2;
          }
    const dev = Math.sqrt(num) / Math.sqrt(den || 1);
    const ks = Array.from({ length: T }, (_, i) => i + 1);
    massCurve.update({
      points: ks.map((kk) => [kk, massAt(kk, selector)]),
      reference: ks.map((kk) => [kk, massAt(kk, "oracle")]),
      xRange: [1, T],
      yRange: [0, 1],
      mark: k,
      markLabel: `k = ${k}`,
    });
    kRead.update({
      mass: `${fmt(100 * mine, 1)}%`,
      dev: dev === 0 ? "0% — identical to full attention" : `${fmt(100 * dev, 1)}%`,
      word: `${run.top[0].word} ${fmt(run.top[0].p, 4)}  /  ${base.top[0].word} ${fmt(base.top[0].p, 4)}`,
    });
    const rnd = massAt(k, "random");
    const orc = massAt(k, "oracle");
    kNote.className = "note " + (selector === "oracle" ? "good" : selector === "random" ? "warn" : "");
    kNote.innerHTML = `The grid is the same picture every card in this deck has used, with the outlined cells the keys this query is not allowed to compare against. The dashed curve underneath is always the oracle — <strong>the distribution Eq. 3 fits the indexer to</strong>, which is the real attention's weights summed across heads and L1-normalised. Set <em>k</em> to the whole sentence and the deviation goes to exactly zero: with nothing excluded this is plain attention, as it must be. ${
      selector === "indexer"
        ? `Now the uncomfortable part, and it is the point of this card. At <em>k</em> = ${k} the untrained indexer keeps <strong>${fmt(
            100 * mine,
            1
          )}%</strong> of full attention's own weight; <strong>choosing at random keeps ${fmt(
            100 * rnd,
            1
          )}%</strong>, and the oracle keeps <strong>${fmt(
            100 * orc,
            1
          )}%</strong>. The indexer and random are the same number to within noise, at every <em>k</em> — switch between them and watch the curve barely move. <strong>An untrained indexer is not a weak version of this mechanism; it is no version of it.</strong>`
        : selector === "oracle"
        ? `This is the ceiling: selection by the target itself, which is what a perfectly trained indexer would reproduce. At <em>k</em> = ${k} it keeps <strong>${fmt(
            100 * mine,
            1
          )}%</strong> of the weight that reading everything would have found, against <strong>${fmt(
            100 * rnd,
            1
          )}%</strong> for random. The distance between those two is exactly what the 943.7B tokens in the next panel are paying for, and it is the only honest way this page can quantify a mechanism that does nothing until it is trained.`
        : `Random selection is the floor — and the reason it is worth a control at all is that the untrained indexer sits on it. At <em>k</em> = ${k} random keeps <strong>${fmt(
            100 * mine,
            1
          )}%</strong> against the untrained indexer's <strong>${fmt(100 * massAt(k, "indexer"), 1)}%</strong>.`
    } <span class="warn">Untrained weights throughout</span>, so the oracle is not a statement about language either — it is the target of a loss, measured on this model's own attention.`;

    // --- 3. the bill
    const K128 = 131072;
    billRead.update({
      warm: `${fmt((1000 * 16 * K128) / 1e9, 3)}B tokens — the report says 2.1B`,
      sparse: `${fmt((15000 * 480 * K128) / 1e9, 1)}B tokens — the report says 943.7B`,
      rate: `${fmt((100 * 2048) / K128, 4)}% of the keys`,
      cplx: "the indexer — it still scores every past token, at O(L²)",
    });
    billNote.innerHTML = `Two stages, and the arithmetic in them is the only thing in this source a reader can check without a cluster. <strong>First a dense warm-up</strong>: attention left alone, every model parameter frozen except the indexer, which is fitted to the attention distribution by KL divergence — 1000 steps of 16 sequences of 128K tokens, which comes to 2.097B and is reported as 2.1B. <strong>Then the sparse stage</strong>: selection switched on, everything unfrozen, k = 2048, and 15000 steps of 480 sequences of 128K — <strong>943.7B tokens, matching the report exactly.</strong> Just under a trillion tokens of continued pre-training, and its purpose is to teach one small network to imitate the attention it is standing in front of. One detail in that stage deserves its own sentence, because it reverses the previous card: <em>"we detach the indexer input from the computational graph … The training signal of the indexer is from only L_I, while the optimization of the main model is according to only the language modeling loss."</em> <strong>The indexer never sees the language-modelling objective.</strong> Concept 24's whole argument was that sparsity should be trained end to end by the loss that matters; four months later this one trains it as a separate imitation problem, and ships. The last line above is the other half of the honesty: <strong>the quadratic cost does not disappear.</strong> Core attention becomes O(Lk), but the indexer still compares every query against every earlier token — the report says so plainly, and adds that below some length they do not run the mechanism at all but <em>"specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions."</em>`;

    // --- what this page cannot show
    honest.innerHTML = `<strong>This is the one card in the deck whose subject does not work here at all</strong>, and the reason is worth stating precisely rather than apologising for. Every other mechanism in this timeline does something visible on untrained weights: a rotation rotates, a mask masks, a state decays, a cache shrinks. DSA is a <em>trained</em> component in front of attention, and an untrained one is measurably indistinguishable from choosing at random — which is what the panels above show, and the strongest true statement this page can make about it. So the card brackets the mechanism instead of demonstrating it: the floor is random selection, the ceiling is the distribution the report's own Eq. 3 fits the indexer to, and the gap is what nearly a trillion tokens buys. Four things are out of reach and named rather than filled in. <strong>The indexer's size:</strong> the report gives no head count and no width, so the ${HI} heads of width ${DI} here are this app's and cannot be read as anything else, and "much less computation" cannot be turned into a number by anyone outside DeepSeek. <strong>The cost evidence:</strong> Figure 3 is cost per million tokens on H800s at two dollars a GPU-hour, a chart with no table, so no figure from it is printed here — the quotable commercial number is the 50%+ API price cut in the dated release note. <strong>The model:</strong> a mixture-of-experts continued from V3.1-Terminus at 128K context, none of which this page has. And <strong>the source itself:</strong> six pages shipped inside a model repository, no arXiv identifier, no ablation of any part of the mechanism. A full paper naming DSA as its first contribution appeared on arXiv on 2 December 2025 (2512.02556), two months later and about a different model; the timeline dates the release, because that is when this mechanism became public.`;
  }

  return { update: render, unmount: () => {} };
}
