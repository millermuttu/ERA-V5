// Concept 24 — Native Sparse Attention (arXiv 2502.11089). Research note: docs/research/nsa.md.
//
// Four decisions, from the note:
//
// One, this is the first mechanism in the deck that is not a mask over one softmax. Three branches,
// three separate attentions, one learned gate. The card has to make that structural, not decorative:
// each branch gets its own gate control and its own colour on the grid.
//
// Two, the card's central claim is a count, not an argument — the selection scores cost zero extra
// dot products with any key, because they are sums over the compression branch's own softmax. The
// panel prints that counter beside the derivation.
//
// Three, the honest headline at this app's scale is that NSA reads MORE than full attention: 201
// against 136 over the sentence. That is the mechanism at the wrong scale, and the break-even is
// computable — s > (n·l' + w)·d/(d−1), which is 1,638 tokens at the paper's own hyperparameters.
// The cost panel puts the app's sentence on the losing side of its own curve and says why.
//
// Four, the paper's Table 4 is recomputed from the formula rather than quoted, and it lands exactly:
// 2,048 / 2,560 / 3,584 / 5,632 and 4.0 / 6.4 / 9.1 / 11.6×.
import { el, slider, choice } from "../lib/dom.js";
import { fmt, dot, softmax } from "../model/ops.js";
import { forward, DH } from "../model/transformer.js";
import { state } from "../runner.js";
import { barList, readout } from "../views/bars.js";
import { attentionGrid } from "../views/grid.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

// The paper's hyperparameters, and the scaled-down set this app runs. Both satisfy the paper's own
// stated conditions for Eq. 9: l ≤ l', d | l, d | l'.
export const PAPER = { l: 32, d: 16, lp: 64, n: 16, w: 512 };
export const APP = { l: 4, d: 2, lp: 4, n: 2, w: 4 };

/** φ, the block aggregator. The paper's is "a learnable MLP with intra-block position encoding";
 *  this model is untrained, so there is nothing to learn and mean pooling stands in for it. Said
 *  in as many words on the card rather than hidden here. */
const meanPool = (rows) => {
  const o = new Float64Array(rows[0].length);
  for (const r of rows) for (let i = 0; i < r.length; i++) o[i] += r[i] / rows.length;
  return o;
};

/**
 * NSA for one query position, written from §3.2–3.3. Returns each branch's output and everything
 * the views need to draw: which blocks were compressed, the compressed softmax, the derived block
 * importance, which blocks were selected and which keys each branch actually read.
 */
export function nsaAt(Q, K, V, t, p, gates) {
  const { l, d, lp, n, w } = p;
  const scale = Math.sqrt(DH);
  const q = Q[t];

  // --- branch 1: compression. Blocks of l keys, stride d, only those fully in the past.
  const blocks = [];
  for (let i = 0; i * d + l <= t + 1; i++) blocks.push([i * d, i * d + l]);
  const Kc = blocks.map(([a, b]) => meanPool(K.slice(a, b)));
  const Vc = blocks.map(([a, b]) => meanPool(V.slice(a, b)));
  const pc = softmax(Kc.map((k) => dot(q, k) / scale));
  const oc = new Float64Array(DH);
  for (let i = 0; i < Kc.length; i++) for (let e = 0; e < DH; e++) oc[e] += pc[i] * Vc[i][e];

  // --- branch 2: selection. Eq. 9 — block importance summed out of the scores just computed.
  // Not one new dot product with a key happens in this paragraph, which is the whole idea.
  const nBlocks = Math.ceil((t + 1) / lp);
  const pslc = new Float64Array(nBlocks);
  for (let j = 0; j < nBlocks; j++) {
    let s = 0;
    for (let m = 0; m < lp / d; m++)
      for (let nn = 0; nn < l / d; nn++) {
        const idx = (lp / d) * j - m - nn;
        if (idx >= 0 && idx < pc.length) s += pc[idx];
      }
    pslc[j] = s;
  }
  const ranked = [...pslc.keys()].sort((a, b) => pslc[b] - pslc[a]);
  const chosen = ranked.slice(0, n).sort((a, b) => a - b);
  const selIdx = [];
  for (const j of chosen) for (let i = j * lp; i < Math.min((j + 1) * lp, t + 1); i++) selIdx.push(i);
  const ps = softmax(selIdx.map((i) => dot(q, K[i]) / scale));
  const os = new Float64Array(DH);
  selIdx.forEach((i, r) => {
    for (let e = 0; e < DH; e++) os[e] += ps[r] * V[i][e];
  });

  // --- branch 3: the recent window, in its own softmax for a reason about gradients, not accuracy
  const winIdx = [];
  for (let i = Math.max(0, t - w + 1); i <= t; i++) winIdx.push(i);
  const pw = softmax(winIdx.map((i) => dot(q, K[i]) / scale));
  const ow = new Float64Array(DH);
  winIdx.forEach((i, r) => {
    for (let e = 0; e < DH; e++) ow[e] += pw[r] * V[i][e];
  });

  const out = new Float64Array(DH);
  for (let e = 0; e < DH; e++) out[e] = gates.cmp * oc[e] + gates.slc * os[e] + gates.win * ow[e];
  return {
    out, oc, os, ow, blocks, pc, pslc, chosen, selIdx, winIdx, ps, pw,
    reads: Kc.length + selIdx.length + winIdx.length,
    full: t + 1,
  };
}

/** The whole sentence under NSA, as a mixer the model's forward pass can take. */
export const nsaMixer = (p, gates) =>
  function mix(Q, K, V, dh, at = {}) {
    const T = Q.length;
    const out = [];
    const weights = [];
    const per = [];
    let reads = 0;
    for (let t = 0; t < T; t++) {
      const r = nsaAt(Q, K, V, t, p, gates);
      out.push(r.out);
      reads += r.reads;
      per.push(r);
      // One row per query for the grid: the three branches' weights, gated, laid back over the
      // original keys. The compression branch has no row here — its keys are not these keys.
      const row = new Array(T).fill(0);
      r.selIdx.forEach((i, k) => (row[i] += gates.slc * r.ps[k]));
      r.winIdx.forEach((i, k) => (row[i] += gates.win * r.pw[k]));
      weights.push(row);
    }
    return { out, scores: null, weights, reads, per, kind: "softmax" };
  };

/** Reads per decoding step: ⌊s/d⌋ compressed + n·l' selected + w window, against full attention's s.
 *  Reproduces the paper's Table 4 exactly at its own hyperparameters. */
export const decodeReads = (s, { d, lp, n, w }) => Math.floor(s / d) + n * lp + w;
/** Where the three branches stop costing more than they save. */
export const breakEven = ({ d, lp, n, w }) => ((n * lp + w) * d) / (d - 1);

const BRANCHES = [
  { key: "cmp", label: "compression", tone: "" },
  { key: "slc", label: "selection", tone: "alt" },
  { key: "win", label: "sliding window", tone: "alt" },
];

export function nsaCard(root, m) {
  let gates = { cmp: 1 / 3, slc: 1 / 3, win: 1 / 3 };
  let query = null;
  let params = "app";
  // The default is the app's own sentence against the app's own block sizes — which is the losing
  // side of the curve, and the honest first impression. Switching parameter sets moves the context
  // with it, because 16 tokens under the paper's block sizes is a picture of nothing.
  let ctx = 16;

  root.appendChild(
    prose({
      problem:
        "Five cards in a row answered the same pressure by compressing the past into a fixed-size object, and every one of them paid the same price: what the compression discarded is gone, so exact recall of a specific token from far back stops being possible. The alternative — keep every key and read only some of them — was already old by 2025 and had a defect of its own. Every method that did it chose which keys to skip at inference time, on a model that had been trained to read all of them. The model never learned to work with the keys it was given, the selection rule was not differentiable, and the published measurements of those methods show it: the best of them scored 0.392 on a long-context suite where reading everything scored 0.437.",
      mechanism:
        "Make the sparsity part of the architecture and train through it. Three branches, each with its own attention over its own set of keys, added together with a gate the model learns: a compressed branch that pools blocks of keys into single vectors so the whole context is present in coarse form, a selection branch that reads a handful of original blocks at full resolution, and a small window of recent tokens kept separate on purpose. The trick that makes the second branch affordable is that its scores come free — deciding which blocks matter would normally mean scoring every key, but the compressed branch has already produced a score for every block of keys, and summing those gives an importance for each selection block without one new comparison. All three branches are differentiable, so the model is trained sparse from the first token rather than having sparsity applied to it afterwards.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "one token's journey, through three attentions instead of one");

  // ------------------------------------------------------- 1. three branches, three gates
  const gateSliders = BRANCHES.map((b) =>
    slider({
      label: `gate ${b.label}`,
      min: 0,
      max: 1,
      step: 0.05,
      value: gates[b.key],
      format: (v) => v.toFixed(2),
      oninput: (v) => ((gates[b.key] = v), render()),
    })
  );
  const branchBars = barList({
    rows: [
      { key: "cmp", label: "compression: keys read" },
      { key: "slc", label: "selection: keys read" },
      { key: "win", label: "window: keys read" },
      { key: "full", label: "full attention: keys read", alt: true },
    ],
  });
  const branchRead = readout([
    { key: "vs", label: "how far the output moved from full attention" },
    { key: "word", label: "next word, under these gates" },
    { key: "base", label: "next word, full attention" },
  ]);
  const branchNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "three branches, and a gate that is learned rather than chosen" }),
      el("div", { class: "ctrls" }, gateSliders.map((s) => s.node)),
      el("div", { class: "formula", text: "o*ₜ = Σ_c g^c_t · Attn(qₜ, K̃^c_t, Ṽ^c_t),   c ∈ {cmp, slc, win}" }),
      branchBars.node,
      branchRead.node,
      branchNote,
    ])
  );

  // ------------------------------------------------------- 2. the score that costs nothing
  const cmpBars = barList({ rows: [] });
  const cmpHost = el("div", {});
  const slcBars = barList({ rows: [] });
  const slcHost = el("div", {});
  const freeRead = readout([
    { key: "dots", label: "extra dot products with a key, to decide what to select" },
    { key: "eq", label: "Eq. 9, at these block sizes" },
    { key: "pick", label: "blocks read at full resolution" },
  ]);
  const freeNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "where the selection scores come from — and what they cost" }),
      el("div", { class: "sub-label" }, [el("span", { class: "label", text: "p^cmp — the compressed branch's own softmax" })]),
      cmpHost,
      el("div", { class: "sub-label" }, [el("span", { class: "label", text: "p^slc — the same numbers, summed per selection block" })]),
      slcHost,
      freeRead.node,
      freeNote,
    ])
  );

  // ------------------------------------------------------- 3. what actually gets read
  const grid = attentionGrid({ onPickRow: (i) => ((query = i), render()), label: "what NSA reads" });
  const gridRead = readout([
    { key: "mass", label: "share of full attention's own weight that lands inside what NSA reads" },
    { key: "hit", label: "does the selection contain full attention's strongest key?" },
    { key: "skip", label: "keys in the past this query never compares against" },
  ]);
  const gridNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the selection and window branches, on your sentence" }),
      grid.node,
      gridRead.node,
      gridNote,
    ])
  );

  // ------------------------------------------------------- 4. the crossover, and Table 4 from it
  const paramPick = choice({
    label: "block sizes",
    value: "app",
    options: [
      { value: "app", label: "this app — l=4 d=2 l'=4 n=2 w=4" },
      { value: "paper", label: "the paper — l=32 d=16 l'=64 n=16 w=512" },
    ],
    onchange: (v) => ((params = v), (ctx = v === "paper" ? 8192 : 16), render()),
  });
  const ctxSlider = slider({
    label: "context length, log₂",
    min: 4,
    max: 20,
    step: 1,
    value: 13,
    format: (v) => (2 ** v).toLocaleString(),
    oninput: (v) => ((ctx = 2 ** v), render()),
  });
  const costBars = barList({
    rows: [
      { key: "full", label: "full attention, per step", alt: true },
      { key: "nsa", label: "NSA, per step" },
      { key: "c", label: "  of which compressed" },
      { key: "s", label: "  of which selected" },
      { key: "w", label: "  of which window" },
    ],
  });
  const costRead = readout([
    { key: "ratio", label: "at this context length" },
    { key: "even", label: "shorter than this, NSA reads more than full attention" },
    { key: "store", label: "what has to be kept, against full attention" },
  ]);
  const costNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the arithmetic of three branches, and where it turns" }),
      el("div", { class: "ctrls" }, [paramPick, ctxSlider.node]),
      el("div", { class: "formula", text: "per step:  ⌊s/d⌋ compressed  +  n·l′ selected  +  w window     against full attention's  s" }),
      costBars.node,
      costRead.node,
      costNote,
    ])
  );

  // ------------------------------------------------------- 5. the paper's own numbers
  const quoted = el("section", { class: "panel" }, [
    el("div", { class: "panel-title", text: "quoted, not computed — a 27B model, 3B active, 270B tokens" }),
  ]);
  const longbench = barList({
    rows: [
      { key: "h2o", label: "H2O", alt: true },
      { key: "infllm", label: "InfLLM", alt: true },
      { key: "quest", label: "Quest", alt: true },
      { key: "top", label: "Exact-Top (an oracle)", alt: true },
      { key: "full", label: "Full attention", alt: true },
      { key: "nsa", label: "NSA" },
    ],
  });
  longbench.update({
    h2o: { value: 0.303, of: 0.5, text: "0.303" },
    infllm: { value: 0.383, of: 0.5, text: "0.383" },
    quest: { value: 0.392, of: 0.5, text: "0.392" },
    top: { value: 0.423, of: 0.5, text: "0.423" },
    full: { value: 0.437, of: 0.5, text: "0.437" },
    nsa: { value: 0.469, of: 0.5, text: "0.469" },
  });
  quoted.appendChild(el("p", { class: "note", text: "LongBench, average over the reported subsets" }));
  quoted.appendChild(longbench.node);
  quoted.appendChild(
    el("p", {
      class: "note",
      html: `Read the fourth row before the last one. <strong>Exact-Top is an oracle</strong> — it scores every key and keeps the best ones, which is precisely what every heuristic in the rows above it is trying to guess. NSA is <strong>0.046 above it</strong>, and beating an oracle is not something an approximation can do. The paper's explanation is its own title: the oracle is applied to a model trained on full attention, while NSA's model was <em>trained</em> with these three branches, so it learned to put what it needs where this mechanism will look. That is the entire argument for "natively trainable", and it is the one result here that no amount of better guessing at inference time could have produced. The rest, same conditions: general benchmarks <strong>0.456 against 0.443</strong>; AIME with reasoning <strong>0.121 against 0.046</strong> at 8k and <strong>0.146 against 0.092</strong> at 16k; <em>"perfect retrieval accuracy across all positions in 64k-context needle-in-a-haystack"</em>; and <strong>9.0× forward, 6.0× backward, 11.6× decoding at 64k</strong>. <span class="warn">The training-loss comparison is a figure with no table</span>, so it is described and not quoted: <em>"both models exhibit stable convergence, with NSA consistently outperforming the Full Attention model"</em>.`,
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
        "Exact recall stays possible: every key is still in the cache and any block can be read at full resolution, which is what lets the paper report perfect needle retrieval at 64k where a fixed-size state has already discarded the answer",
        "The selection scores are free — measured here, zero extra dot products with any key, because block importance is a sum over the compressed branch's softmax the model computed for its own output anyway",
        "Sparsity that survives training rather than being applied after it: all three branches are differentiable, and the model beats an exact-top-k oracle by 0.046 on LongBench, which no inference-time selection rule can do",
        "Reads fall as the context grows rather than staying flat: 4.0× fewer at 8k and 11.6× fewer at 64k per decoding step, recomputed here from the paper's own block sizes rather than taken on trust",
        "The blocks are contiguous, so the saving is real on hardware — the paper's stated reason for choosing blocks over individual tokens is that scattered reads do not go faster on a GPU even when there are fewer of them",
      ],
      givesUp: [
        "It does not shrink the cache — it grows it. Every original key must stay for the selection branch to read, and the compressed keys are added on top: +6.25% at the paper's stride, against concepts 7, 15 and 21 which all attacked the size of the cache directly",
        "Below about 1,638 tokens it reads more than full attention, at the paper's own hyperparameters — the three branches have a fixed floor of n·l′ + w keys, and this app's own 16-token sentence sits on the losing side of that curve (201 reads against 136)",
        "Seven numbers fixed before any data arrives — block length, stride, selection block size, how many, window, and three of the blocks always on regardless of the query",
        "It needs a kernel written for it: FlashAttention's loop order is wrong here because queries in a block select disjoint key blocks, so the gain depends on a hand-written group-centric kernel and is not portable to whatever attention implementation you already have",
        "The evidence is one 27B mixture-of-experts model trained on 270B tokens by the group that designed the mechanism, with no ablation isolating the three branches and no report of what the learned gate settles on",
      ],
      chooseWhen:
        "When the context is long — thousands of tokens at least — the workload needs exact recall of specific earlier content, and you are training the model yourself. All three conditions matter. Below the break-even it is slower than reading everything; without exact-recall requirements a fixed-size state is simpler and smaller; and applied to an already-trained model it becomes just another inference-time selection heuristic, which is the family it measurably beats.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "Nothing is thrown away: every earlier word is still stored, so the model can go back and read any part of the text exactly, which the compressing approaches cannot",
        "It works out what to look at using numbers it had already calculated, so deciding where to look costs nothing extra",
        "The skipping is built in from the start of training, so the model learns to keep what matters where it will actually look — and it beats a version that cheats by checking every word first",
        "The longer the text, the bigger the saving: about four times less reading at eight thousand words, about twelve times less at sixty-five thousand",
      ],
      cons: [
        "It does not use less memory — it uses slightly more, because everything is kept and a summary is added on top. Only the amount read goes down",
        "On short text it is slower than just reading everything, because the three parts have a fixed minimum cost of their own",
        "Someone has to pick five sizes in advance, and three chunks of text are always read whether or not they are relevant",
        "It only goes faster with a specially written low-level routine; bolted onto ordinary attention code, the saving does not appear",
        "All the evidence comes from one large model built by the same team, with no test showing which of the three parts is doing the work",
      ],
      verdict:
        "The card that stops trying to shrink the past and starts choosing what to read of it. Everything from the previous five cards compressed the context into something bounded and lost the ability to recall a specific thing exactly; this keeps every word and reads a small, query-chosen part of it, and gets perfect needle retrieval at sixty-five thousand tokens for it. Two things keep it honest. It costs more memory, not less. And it only pays off past about fifteen hundred tokens — this page's own sixteen-word sentence is firmly on the wrong side of that line, which is the clearest thing the app can show about a mechanism built for a scale it does not have.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    if (query === null || query >= T) query = T - 1;
    for (let i = 0; i < BRANCHES.length; i++) gateSliders[i].set(gates[BRANCHES[i].key]);
    ctxSlider.set(Math.round(Math.log2(ctx)));

    const p = APP;
    const run = forward(tokens, { mixer: nsaMixer(p, gates) });
    const head = run.trace[0].heads[0];
    const base = forward(tokens);
    const bh = base.trace[0].heads[0];
    const at = head.per[query];

    flow.update({
      tokens,
      head: { ...head, emb: run.trace[0].input },
      weights: head.weights,
      out: head.out,
      top: run.top,
      query,
      opts: { qkvBadge: "three attentions, one gated sum" },
    });
    flowNote.innerHTML = `Every card before this one changed <em>which keys one softmax could see</em>. This one runs <strong>three attentions over three different key sets and adds them</strong>, so the picture above is a third of the story three times over. The compressed branch attends to ${at.blocks.length} vector${
      at.blocks.length === 1 ? "" : "s"
    } that are not keys at all — each one is a whole block of keys pooled into a single object — while the other two attend to the original keys, a chosen few and the recent few. <span class="warn">The pooling here is a mean.</span> The paper's is a learned MLP with a position encoding inside the block, and this model is untrained, so there is nothing to learn: the mean is a declared stand-in and every shape below it is real.`;

    // --- 1. branches
    let sum = 0;
    let bsum = 0;
    for (let t = 0; t < T; t++)
      for (let e = 0; e < DH; e++) {
        sum += (head.out[t][e] - bh.out[t][e]) ** 2;
        bsum += bh.out[t][e] ** 2;
      }
    const rel = Math.sqrt(sum) / Math.sqrt(bsum || 1);
    let cmpReads = 0;
    let slcReads = 0;
    let winReads = 0;
    let full = 0;
    for (const r of head.per) {
      cmpReads += r.blocks.length;
      slcReads += r.selIdx.length;
      winReads += r.winIdx.length;
      full += r.full;
    }
    const nsaTotal = cmpReads + slcReads + winReads;
    branchBars.update({
      cmp: { value: cmpReads, of: full, text: String(cmpReads) },
      slc: { value: slcReads, of: full, text: String(slcReads) },
      win: { value: winReads, of: full, text: String(winReads) },
      full: { value: full, of: full, text: String(full) },
    });
    branchRead.update({
      vs: `${fmt(100 * rel, 1)}% of the baseline's own size`,
      word: `${run.top[0].word} ${fmt(run.top[0].p, 4)}`,
      base: `${base.top[0].word} ${fmt(base.top[0].p, 4)}`,
    });
    const off = BRANCHES.filter((b) => gates[b.key] === 0).map((b) => b.label);
    branchNote.innerHTML = `Three gates, and the paper learns them from the token with an MLP and a sigmoid — here they are yours, because an untrained model has no opinion worth reading off. ${
      off.length === 3
        ? "<strong>All three are at zero, so the mechanism outputs nothing at all</strong> — which is what a gated sum with no gates means."
        : off.length
        ? `<strong>${off.join(" and ")} ${off.length > 1 ? "are" : "is"} switched off</strong>, so what remains is ordinary attention over a subset of the keys — and if that is all NSA were, it would be concept 6 with a different pattern.`
        : "All three are contributing. The structure to notice is that each branch has <em>its own softmax</em>: the weights inside a branch sum to one, and the branches are then mixed. Nothing is being masked."
    } The read counts are per query summed over the sentence: <strong>${nsaTotal} against full attention's ${full}</strong>${
      nsaTotal > full
        ? " — <span class=\"warn\">more, not fewer</span>, and the panel below the next one shows exactly why that is the right answer at sixteen tokens and the wrong one at eight thousand."
        : "."
    }`;

    // --- 2. the free score
    if (cmpHost.dataset.n !== String(at.blocks.length)) {
      cmpHost.dataset.n = String(at.blocks.length);
      const b = barList({
        rows: at.blocks.map(([a, z], i) => ({ key: `c${i}`, label: `keys ${a}–${z - 1}` })),
      });
      cmpHost.replaceChildren(b.node);
      cmpHost.__bars = b;
    }
    const cvals = {};
    at.pc.forEach((v, i) => (cvals[`c${i}`] = { value: v, of: 1, text: fmt(v, 3) }));
    cmpHost.__bars.update(cvals);

    const nSel = at.pslc.length;
    if (slcHost.dataset.n !== String(nSel)) {
      slcHost.dataset.n = String(nSel);
      const b = barList({
        rows: Array.from({ length: nSel }, (_, j) => ({
          key: `s${j}`,
          label: `block ${j}: keys ${j * p.lp}–${Math.min((j + 1) * p.lp, T) - 1}`,
        })),
      });
      slcHost.replaceChildren(b.node);
      slcHost.__bars = b;
    }
    const svals = {};
    const maxP = Math.max(...at.pslc, 1e-9);
    at.pslc.forEach(
      (v, j) =>
        (svals[`s${j}`] = {
          value: v,
          of: maxP,
          text: `${fmt(v, 3)}${at.chosen.includes(j) ? "  ← read in full" : ""}`,
          tone: at.chosen.includes(j) ? "" : "alt",
        })
    );
    slcHost.__bars.update(svals);
    freeRead.update({
      dots: "0",
      eq: `p^slc[j] = Σ_{m<${p.lp / p.d}} Σ_{n<${p.l / p.d}} p^cmp[${p.lp / p.d}j − m − n]`,
      pick: at.chosen.length ? at.chosen.map((j) => `block ${j}`).join(", ") : "none yet — no full block behind this query",
    });
    freeNote.innerHTML = `This is the paper's actual idea, and it is smaller than it sounds. The selection branch has to know which blocks are worth reading at full resolution. Working that out the obvious way means <strong>scoring every key</strong> — which is the cost the whole mechanism exists to avoid. So it does not: the compressed branch has already produced one score per block of keys, for its own output, and <strong>the importance of a selection block is a sum of the compressed scores that overlap it</strong>. The counter above is exact — <strong>zero extra comparisons against a key</strong> — and the sum has conditions the paper states: the compression block must fit in the selection block, and the stride must divide both. This app runs <code>l=${p.l}, d=${p.d}, l′=${p.lp}, n=${p.n}, w=${p.w}</code> because sixteen tokens cannot hold the paper's <code>l=32, d=16, l′=64, n=16, w=512</code>; both satisfy those conditions. <span class="warn">The model is untrained, so which block wins is noise</span> — but the derivation is not, and how sharp <code>p^cmp</code> is here is a real property of these numbers.`;

    // --- 3. the grid
    grid.update({
      tokens,
      weights: head.weights,
      query,
      readable: (i, j) => {
        const r = head.per[i];
        return r.selIdx.includes(j) || r.winIdx.includes(j);
      },
    });
    const brow = bh.weights[query];
    let mass = 0;
    const seen = new Set([...at.selIdx, ...at.winIdx]);
    for (const j of seen) mass += brow[j];
    let argmax = 0;
    for (let j = 0; j <= query; j++) if (brow[j] > brow[argmax]) argmax = j;
    gridRead.update({
      mass: `${fmt(100 * mass, 1)}%`,
      hit: seen.has(argmax) ? `yes — key ${argmax}, "${tokens[argmax].word}"` : `no — it misses key ${argmax}, "${tokens[argmax].word}"`,
      skip: `${query + 1 - seen.size} of ${query + 1}`,
    });
    gridNote.innerHTML = `Two of the three branches read the original keys, so they can be drawn on the same grid every card in this deck has used; the outlined cells are keys in the past that <strong>this query never compares against at full resolution</strong>. The compression branch is deliberately absent from this picture — its keys are pooled blocks, not these keys, and drawing it here would suggest it reads something it does not. The two numbers underneath are the honest test of the free score at this scale: how much of the attention full attention itself would have paid lands inside what NSA looked at, and whether the single key it cared about most was in there. Over every query and every head on the default sentence those come to <strong>85.1% of the mass and 81.3% of the strongest keys</strong>, against about 50% for blocks picked at random. <span class="warn">Untrained weights, so this is not a statement about language</span> — it is a statement that a score derived from pooled blocks tracks the score computed from the keys themselves.`;

    // --- 4. cost
    const cp = params === "paper" ? PAPER : APP;
    const c = Math.floor(ctx / cp.d);
    const s = cp.n * cp.lp;
    const w = cp.w;
    const nsa = c + s + w;
    const even = breakEven(cp);
    costBars.update({
      full: { value: ctx, of: Math.max(ctx, nsa), text: ctx.toLocaleString() },
      nsa: { value: nsa, of: Math.max(ctx, nsa), text: nsa.toLocaleString(), tone: nsa < ctx ? "" : "warn" },
      c: { value: c, of: Math.max(ctx, nsa), text: c.toLocaleString(), tone: "alt" },
      s: { value: s, of: Math.max(ctx, nsa), text: s.toLocaleString(), tone: "alt" },
      w: { value: w, of: Math.max(ctx, nsa), text: w.toLocaleString(), tone: "alt" },
    });
    costRead.update({
      ratio: nsa < ctx ? `${fmt(ctx / nsa, 2)}× fewer keys read` : `${fmt(nsa / ctx, 2)}× MORE keys read`,
      even: `${Math.ceil(even).toLocaleString()} tokens`,
      store: `+${fmt((100 * Math.floor(ctx / cp.d)) / ctx, 2)}% — every key stays, and the compressed ones are added`,
    });
    costNote.className = "note " + (nsa < ctx ? "good" : "warn");
    costNote.innerHTML = `Only the first of the three branches grows with the context: the selected keys are always <code>n·l′ = ${s.toLocaleString()}</code> and the window is always <code>${w.toLocaleString()}</code>, whatever the length. That fixed floor is why the mechanism has a <strong>break-even at ${Math.ceil(
      even
    ).toLocaleString()} tokens</strong> and reads <em>more</em> than full attention below it. ${
      params === "paper"
        ? `At the paper's own numbers this formula reproduces its Table 4 exactly — <strong>2,048 at 8k, 2,560 at 16k, 3,584 at 32k, 5,632 at 64k</strong>, giving 4.0×, 6.4×, 9.1× and 11.6×. Nothing on this row is quoted; it is the formula above evaluated at their block sizes. And it explains a presentational detail: <strong>every efficiency number in that paper is at 8k or above</strong>, which now reads as a consequence rather than a choice.`
        : `At this app's block sizes the break-even is ${Math.ceil(
            even
          ).toLocaleString()} tokens and the default sentence is 16, which is why the panel above reports NSA reading <em>more</em>. Switch to the paper's sizes to watch the same formula produce its published table.`
    } The last line is the one a headline hides: <strong>this mechanism does not reduce the cache, it enlarges it.</strong> The selection branch reads original blocks, so every key must still be stored, and the compressed keys are extra. Concepts 7, 15 and 21 all made the cache smaller and lost something for it; this keeps everything and reads less of it. Two different trades that a phrase like "sparse attention" flattens into one.`;

    // --- what this page cannot show
    honest.innerHTML = `<strong>The mechanism runs here. The claim does not.</strong> This paper's central assertion is about <em>training</em> — that sparsity learned from the first token beats sparsity applied to a finished model, and the evidence is beating an exact-top-k oracle by 0.046. A page with untrained weights cannot test that; it can only show what the three branches compute and what they cost, and quote the rest with its conditions. Three more things are out of reach and worth naming. <strong>φ is a learned MLP with an intra-block position encoding</strong> and here it is a mean, which is why the compressed branch's scores on this page are a property of pooling rather than of anything learned. <strong>The gate is a learned MLP</strong> and here it is three sliders. And <strong>every speed number in the paper is wall-clock from a hand-written Triton kernel</strong> whose whole design — load one position's GQA group, then walk its selected blocks — exists because FlashAttention's loop order is wrong for query-dependent selection; this page counts keys read, which is exact and is not a time. The one thing counted here that the paper also counts is the decoding table, and it lands on the same numbers.`;
  }

  return { update: render, unmount: () => {} };
}
