// Concept 23 — Gated DeltaNet (arXiv 2412.06464). Research note: docs/research/gated-deltanet.md.
//
// Four decisions, from the note:
//
// One, the card's spine is the paper's own Table 1: every rule in this family is the minimiser of
// a per-token objective, and the two mechanisms are two separate terms in it. α moves the anchor,
// β changes the target. Four rules, two switches, one family — and the seam already spells all
// four, once the add rule is allowed to decay.
//
// Two, the gate is shown as what it measurably is: a bounded memory, not a filter. α multiplies
// the whole state, so it cannot spare one pair. The lab probes the same run at both ends and the
// two directions of the paper's Table 2 appear as one curve tilting.
//
// Three, the gain does not reproduce here and the card says so. At d = 8 with random keys the gate
// adds almost nothing on top of the delta rule; the paper's evidence is 1.3B parameters over 100B
// tokens with a *learned* gate, and it stays in the quoted panel where it belongs.
//
// Four, no chunkwise anything. §3.3 folds the decay into concept 22's WY form and that card
// already demonstrates chunking with an exact equivalence result. This one is about the rule.
import { el, slider, choice, toggle } from "../lib/dom.js";
import { fmt, dot, mulberry32, gauss } from "../model/ops.js";
import { forward, DH } from "../model/transformer.js";
import { stateMixer } from "../model/mixers.js";
import { state } from "../runner.js";
import { barList, readout } from "../views/bars.js";
import { curveView } from "../views/curve.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

// ------------------------------------------------------------------ §3.4's feature map
// "linear projection, short convolution and SiLU, with L2 normalization applied to q, k". The
// convolution has no analogue here and is not faked; the SiLU and the L2 are exactly concept 22's.
const silu = (x) => x / (1 + Math.exp(-x));
const l2 = (v) => {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  return Float64Array.from(v, (x) => x / s);
};
const featureMap = (v) => l2(Float64Array.from(v, silu));

const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Declared stand-ins for the paper's two linear projections of x_t. The mixer is handed keys and
 *  values, never the residual stream — and the model is untrained, so no learned α exists to copy.
 *  `strength` is a knob for exactly that reason: the direction is real, the value is arbitrary. */
const PROJ = (() => {
  const rnd = mulberry32(4242);
  return {
    a: Float64Array.from({ length: DH }, () => gauss(rnd)),
    b: Float64Array.from({ length: DH }, () => gauss(rnd)),
  };
})();
const halfLife = (g) => (g >= 1 ? Infinity : g <= 0 ? 0 : Math.log(0.5) / Math.log(g));

// ----------------------------------------------------------------- Table 1, as configuration
// The four rows the paper tabulates, each one a setting of the same seam. Mamba2's row is
// unreachable from the delta form at any β — the delta write is always tied to the read of the
// same key — which is why the add rule has to be able to decay for this comparison to exist.
const RULES = {
  la: {
    label: "linear attention",
    gate: false,
    delta: false,
    objective: "‖S<sub>t</sub> − S<sub>t−1</sub>‖²<sub>F</sub> − 2⟨S<sub>t</sub>k<sub>t</sub>, v<sub>t</sub>⟩",
    update: "S<sub>t</sub> = S<sub>t−1</sub> + v<sub>t</sub>k<sub>t</sub>ᵀ",
    concept: "concept 9",
  },
  mamba2: {
    label: "Mamba2",
    gate: true,
    delta: false,
    objective: "‖S<sub>t</sub> − <b>α<sub>t</sub></b>S<sub>t−1</sub>‖²<sub>F</sub> − 2⟨S<sub>t</sub>k<sub>t</sub>, v<sub>t</sub>⟩",
    update: "S<sub>t</sub> = <b>α<sub>t</sub></b>S<sub>t−1</sub> + v<sub>t</sub>k<sub>t</sub>ᵀ",
    concept: "concept 20's successor",
  },
  delta: {
    label: "DeltaNet",
    gate: false,
    delta: true,
    objective:
      "‖S<sub>t</sub> − S<sub>t−1</sub>‖²<sub>F</sub> − 2⟨S<sub>t</sub>k<sub>t</sub>, <b>β<sub>t</sub>(v<sub>t</sub> − S<sub>t−1</sub>k<sub>t</sub>)</b>⟩",
    update: "S<sub>t</sub> = S<sub>t−1</sub>(I − <b>β<sub>t</sub></b>k<sub>t</sub>k<sub>t</sub>ᵀ) + <b>β<sub>t</sub></b>v<sub>t</sub>k<sub>t</sub>ᵀ",
    concept: "concepts 11 and 22",
  },
  gated: {
    label: "Gated DeltaNet",
    gate: true,
    delta: true,
    objective:
      "‖S<sub>t</sub> − <b>α<sub>t</sub></b>S<sub>t−1</sub>‖²<sub>F</sub> − 2⟨S<sub>t</sub>k<sub>t</sub>, <b>β<sub>t</sub></b>(v<sub>t</sub> − <b>α<sub>t</sub></b>S<sub>t−1</sub>k<sub>t</sub>)⟩",
    update: "S<sub>t</sub> = S<sub>t−1</sub>(<b>α<sub>t</sub></b>(I − <b>β<sub>t</sub></b>k<sub>t</sub>k<sub>t</sub>ᵀ)) + <b>β<sub>t</sub></b>v<sub>t</sub>k<sub>t</sub>ᵀ",
    concept: "this card",
  },
};

/** One rule as a mixer configuration. α applies only where the rule has a gate, β only where it
 *  has a delta term — so the sliders are inert on the rows that do not carry them, which is the
 *  honest behaviour rather than a hidden reinterpretation. */
export const ruleMixer = (rule, alpha, beta, perToken = false) => {
  const r = RULES[rule];
  const gateOf = perToken
    ? (i, k) => Math.exp(-softplus(dot(k, PROJ.a)) * (1 - alpha) * 4)
    : () => alpha;
  const betaOf = perToken ? (i, g, k) => sigmoid(dot(k, PROJ.b)) : () => beta;
  return stateMixer({
    write: r.delta ? "gated" : "add",
    decay: r.gate ? gateOf : 1,
    beta: r.delta ? betaOf : 1,
    features: featureMap,
    sumNorm: false,
    attnNorm: false,
  });
};

// ------------------------------------------------------------------ the associative-memory lab
// Standalone, and deliberately not the sentence: capacity is a claim about the number of pairs
// written, and the sentence is 16 tokens. The state is d_h x d_h, the model's own head state.
const LAB_N = 64;
const LAB_SEEDS = 16;
const unit = (rnd, d) => l2(Float64Array.from({ length: d }, () => gauss(rnd)));
const cos = (a, b) => dot(a, b) / (Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b)) || 1);

const labPairs = (seed) => {
  const rnd = mulberry32(seed);
  const K = [];
  const V = [];
  for (let i = 0; i < LAB_N; i++) {
    K.push(unit(rnd, DH));
    V.push(unit(rnd, DH));
  }
  return { K, V };
};
const PAIRS = Array.from({ length: LAB_SEEDS }, (_, s) => labPairs(101 + s));

/** Write every pair in order, then ask the finished state for each key in turn. The keys are
 *  already unit vectors, so the feature map is the identity here — φ is the model's business, not
 *  this lab's. Returns mean recall per write index, averaged over the seeds. */
const labMemo = new Map();
function labProfile(rule, alpha, beta) {
  const key = `${rule}|${alpha}|${beta}`;
  if (labMemo.has(key)) return labMemo.get(key);
  const r = RULES[rule];
  const acc = new Float64Array(LAB_N);
  for (const { K, V } of PAIRS) {
    const mix = stateMixer({
      write: r.delta ? "gated" : "add",
      decay: r.gate ? alpha : 1,
      beta: r.delta ? beta : 1,
      features: (v) => v,
      sumNorm: false,
      attnNorm: false,
    });
    const S = mix(K, K, V, DH, {}).state;
    for (let i = 0; i < LAB_N; i++) {
      const got = Float64Array.from(S, (row) => dot(row, K[i]));
      acc[i] += cos(got, V[i]) / LAB_SEEDS;
    }
  }
  labMemo.set(key, acc);
  return acc;
}

export function gatedDeltanetCard(root, m) {
  let rule = "gated";
  let alpha = 0.9;
  let beta = 1;
  let perToken = false;

  root.appendChild(
    prose({
      problem:
        "Two cards ago the state learned to forget on command, and one card ago the rule that corrects what it already wrote finally became trainable. Each is missing what the other has. A state that only decays writes without looking: it adds the new pair on top of whatever was there, so two associations that share a direction end up superimposed and neither can be read back. A state that only corrects never releases anything: it can overwrite one key exactly, but the seventy keys it will never see again sit in the same fixed-size object forever, and a fixed-size object holding everything holds nothing clearly. The published evidence for that split is unusually clean — on the same benchmark suite, the correcting model scores 98.8 where the decaying one scores 30.4, and 14.4 where the decaying one scores 17.0. Opposite directions, one table.",
      mechanism:
        "Write both mechanisms into the same update. Decay the state first, then correct the decayed state towards the new value — one scalar in front of the old state, one scalar on the correction, read independently off the token. The paper's framing is the one worth keeping: every rule in this family is the closed-form answer to a small optimisation problem solved once per token, and the two mechanisms are two separate terms in it. The decay moves what the state is pulled back towards; the write strength changes what it is pulled towards. Nothing else in the recurrence changes, which is why the chunked training algorithm from the previous card survives with the decay folded in as a running product.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "one rule, on your sentence — the same grid, two scalars");

  // ------------------------------------------------------- 1. the family, as one objective
  const rulePick = choice({
    label: "rule",
    value: rule,
    options: Object.entries(RULES).map(([k, v]) => ({ value: k, label: v.label })),
    onchange: (v) => ((rule = v), render()),
  });
  const alphaSlider = slider({
    label: "decay α",
    min: 0.5,
    max: 1,
    step: 0.01,
    value: alpha,
    format: (v) => v.toFixed(2),
    oninput: (v) => ((alpha = v), render()),
  });
  const betaSlider = slider({
    label: "write strength β",
    min: 0,
    max: 1,
    step: 0.05,
    value: beta,
    format: (v) => v.toFixed(2),
    oninput: (v) => ((beta = v), render()),
  });
  const tokenToggle = toggle({
    label: "read α and β off each token",
    value: false,
    onchange: (v) => ((perToken = v), render()),
  });
  const objRow = el("div", { class: "formula" });
  const updRow = el("div", { class: "formula" });
  const famRead = readout([
    { key: "norm", label: "size of the state after your sentence" },
    { key: "err", label: "average error reading back its own pairs" },
    { key: "gates", label: "distinct decay values applied" },
    { key: "word", label: "next word under this rule" },
  ]);
  const famNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "four rules, one objective, two switches" }),
      el("div", { class: "ctrls" }, [rulePick, alphaSlider.node, betaSlider.node, tokenToggle]),
      el("div", { class: "panel-title", text: "what the update is the answer to" }),
      objRow,
      el("div", { class: "panel-title", text: "and the update itself" }),
      updRow,
      famRead.node,
      famNote,
    ])
  );

  // ------------------------------------------------------- 2. what the state remembers, by age
  const profile = curveView({
    xLabel: "pair index — 0 is written first, 63 last",
    yLabel: "recall",
    ariaLabel: "recall of each written pair against how long ago it was written",
  });
  const labBars = barList({
    rows: [
      { key: "first", label: "first pair written" },
      { key: "last", label: "most recent pair" },
      { key: "recent", label: "last four, averaged" },
      { key: "all", label: "all 64, averaged" },
    ],
  });
  const labRead = readout([
    { key: "hl", label: "half-life of the state, in writes" },
    { key: "cross", label: "against the same rule with no decay" },
  ]);
  const labNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "a fixed state, 64 pairs, and the same run probed at both ends" }),
      profile.node,
      labBars.node,
      labRead.node,
      labNote,
    ])
  );

  // ------------------------------------------------------- 3. the corners, measured live
  const cornerRead = readout([
    { key: "one", label: "α = 1 against the DeltaNet card, largest difference" },
    { key: "zero", label: "β = 0, size of the state after your sentence" },
    { key: "exact", label: "β = 1, what the state returns for the newest key" },
    { key: "mamba", label: "Mamba2's row, reachable from the delta form?" },
  ]);
  const cornerNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the corners of the two switches" }),
      cornerRead.node,
      cornerNote,
    ])
  );

  // ------------------------------------------------------- 4. the paper's own evidence, quoted
  // The paper's Table 2, quoted. Bars rather than a table because the deck has no table style and
  // because the bar is the point: the score at the longest length is where the three models part.
  const T2 = [
    {
      task: "S-NIAH-1 — a repeated, information-free haystack. Retention, and nothing else.",
      lengths: "1K · 2K · 4K · 8K",
      rows: [
        ["DeltaNet", [97.4, 96.8, 99.0, 98.8]],
        ["Mamba2", [99.2, 98.8, 65.4, 30.4]],
        ["Gated DeltaNet", [98.4, 88.4, 91.4, 91.8]],
      ],
    },
    {
      task: "S-NIAH-2 — a real-essay haystack, a number to find. Clearance.",
      lengths: "1K · 2K · 4K · 8K",
      rows: [
        ["DeltaNet", [98.4, 45.6, 18.6, 14.4]],
        ["Mamba2", [99.4, 98.8, 56.2, 17.0]],
        ["Gated DeltaNet", [100.0, 99.8, 92.2, 29.6]],
      ],
    },
    {
      task: "S-NIAH-3 — the same, with a UUID to find. Precision of what was written.",
      lengths: "1K · 2K · 4K",
      rows: [
        ["DeltaNet", [85.2, 47.0, 22.4]],
        ["Mamba2", [64.4, 47.6, 4.6]],
        ["Gated DeltaNet", [86.6, 84.2, 27.6]],
      ],
    },
  ];
  const quoted = el("section", { class: "panel" }, [
    el("div", { class: "panel-title", text: "quoted, not computed — the paper's Table 2, in full" }),
  ]);
  for (const t of T2) {
    const bars = barList({ rows: t.rows.map(([name]) => ({ key: name, label: name, alt: name !== "Gated DeltaNet" })) });
    const vals = {};
    for (const [name, xs] of t.rows) {
      vals[name] = { value: xs[xs.length - 1], of: 100, text: xs.map((x) => x.toFixed(1)).join(" · ") };
    }
    bars.update(vals);
    quoted.appendChild(el("p", { class: "note", text: `${t.task}  (${t.lengths})` }));
    quoted.appendChild(bars.node);
  }
  quoted.appendChild(
    el("p", {
      class: "note",
      html: `The bar is the score at the longest length the paper reports for that task; the numbers beside it are every length, shortest first. Conditions, because they are the whole meaning of the figures: <strong>1.3B parameters, 100B tokens of FineWeb-Edu, 4K training length</strong>, needles hidden in filler. Read it across the three tasks rather than down one. <strong>The first is the cost of the gate</strong> — nothing in that haystack needs clearing, and the correcting model holds 98.8 at 8K while the gated one drops to 91.8, because a decay does not know which of the things it fades mattered. <strong>The second is what the gate buys</strong> — real prose as filler, and the correcting model collapses to 14.4 where the gated one holds 29.6, because a state with no release valve becomes a superposition of everything it has seen. <strong>The third is the delta rule's own column</strong>: a UUID instead of a number, 4.6 for the purely decaying model against 27.6. <span class="warn">Every model collapses at 8K on the second task</span>, and the third stops at 4K. And the paper offers no ablation isolating α from β: this is a comparison of three models, not of three terms — Appendix B.2's ablation table could not be reached for the research note.`,
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
        "Two independent controls over one fixed-size memory, where every earlier member of this family had one: what fraction of the past survives this token, and how hard the current pair is written into what survives",
        "The delta rule's exact overwrite is kept — at full write strength and with the paper's own L2-normalised keys, the state returns the current value for the current key exactly, whatever the decay did a moment earlier (measured here: cosine 1.000 at every decay setting)",
        "The gate turns an accumulating state into a bounded one: in this app's own lab the add rule's most recent pair reads back at 0.34 with no decay and 0.91 at α = 0.8",
        "On the paper's needle suite, both directions improve over the model that is missing the other half — 91.8 against Mamba2's 30.4 where retention matters, 29.6 against DeltaNet's 14.4 where clearance does",
        "It costs nothing to train relative to the previous card: the decay folds into the same chunked WY form as a running product, so the linear-time chunkwise algorithm carries over unchanged",
        "The hybrids that add sliding-window attention on top reach 40.1 on real-world recall against a full-attention transformer's 37.0 — the first row in this family's history to pass it",
      ],
      givesUp: [
        "Retention, and the paper's own table prints the price: 98.8 → 91.8 on the task where nothing needs clearing. A decay that multiplies the whole state cannot spare the one pair that mattered",
        "The state is still one fixed object, and no setting of either switch changes its capacity — in this app's lab the average recall over 64 pairs falls monotonically as the decay strengthens, 0.125 → 0.085, because forgetting is not compression",
        "Pure recurrence still loses to full attention on real-world recall, 30.6 against 37.0, and the hybrids only pass it by putting attention back into the stack",
        "The perplexity gain over the model it improves on is 0.14 — the two mechanisms buy recall, not language modelling, and a benchmark table that reports only perplexity would show almost nothing",
        "Two new scalars per token per head are two more things to learn, and the paper inherits the parameterisation of one of them from another paper without restating it",
        "The evidence separating the two mechanisms is a comparison of three trained models, not an ablation of two terms; nothing in the paper isolates the gate from the delta rule inside one model",
      ],
      chooseWhen:
        "When a fixed-size recurrent state is the right shape for the workload — long streams, constant memory per token, no growing cache — and the stream contains both things worth holding onto and long stretches worth dropping. That combination is what the two switches are for, and it is most of real text. Not when the task is exact recall of arbitrary detail from far back: this improves a bounded memory, it does not remove the bound, and the paper's own hybrids concede the point by adding attention layers back.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The model gets two separate dials instead of one: how much of what it already knew to keep, and how firmly to write down what it is reading now",
        "It can still go back and correct something it recorded wrongly, which the forgetting-only version could never do",
        "It can still let go of a long stretch of text that turned out not to matter, which the correcting-only version could never do",
        "The most recent thing it was told is stored perfectly, no matter how much it has just forgotten",
        "It costs no more to train than the version it improves on, and mixing a few ordinary attention layers in makes it better than a standard model at looking things up",
      ],
      cons: [
        "Forgetting is all-or-nothing across the whole memory: it cannot fade the noise and keep the one sentence that mattered, so on text where nothing should be forgotten it does worse than the simpler version",
        "The memory is still one fixed-size object, so past a certain amount of material things blur together however the dials are set",
        "On looking up specific details it is still behind an ordinary attention model, and the versions that catch up do it by adding attention back in",
        "It writes better sentences by only the slimmest margin — the gain is in remembering, not in fluency",
        "Nobody has shown which of the two dials is doing the work; the evidence is three whole models compared, not one model with a part removed",
      ],
      verdict:
        "The card where the two halves of this family stop being alternatives. One line of arithmetic decays the memory and then corrects it, and the paper's own table shows each half covering exactly where the other fails: the forgetting model dies on the task that needs retention, the correcting model dies on the task that needs clearance, and the combination is second-best on both instead of best on one and hopeless on the other. What it does not do is remove the bound. A fixed-size memory with better manners is still a fixed-size memory — which is why the next card stops trying to compress the past at all and goes back to keeping every key, while choosing which few to read.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    const r = RULES[rule];
    alphaSlider.set(alpha);
    betaSlider.set(beta);

    // --- 1. the family
    objRow.innerHTML = r.objective;
    updRow.innerHTML = r.update;
    const run = forward(tokens, { mixer: ruleMixer(rule, alpha, beta, perToken) });
    const head = run.trace[0].heads[0];
    const S = head.state;
    let frob = 0;
    for (const row of S) for (const x of row) frob += x * x;
    frob = Math.sqrt(frob);
    // Read the finished state back with each key it was written with. Relative, so the number is
    // comparable across rules whose states differ in size by two orders of magnitude.
    let err = 0;
    for (let j = 0; j < T; j++) {
      const k = featureMap(head.K[j]);
      let e = 0;
      let n = 0;
      for (let a = 0; a < DH; a++) {
        const got = dot(S[a], k);
        e += (got - head.V[j][a]) ** 2;
        n += head.V[j][a] ** 2;
      }
      err += Math.sqrt(e) / Math.sqrt(n || 1) / T;
    }
    const gates = head.gates.length ? head.gates : new Array(T).fill(1);
    const distinct = new Set(gates.slice(0, T).map((g) => g.toFixed(9))).size;

    flow.update({
      tokens,
      head: { ...head, emb: run.trace[0].input },
      weights: null,
      out: head.out,
      top: run.top,
      query: T - 1,
      opts: {
        stateMode: { matrix: head.snapshots[T - 1] },
        qkvBadge: perToken ? "α and β read off each token" : "α and β constant across the sentence",
      },
    });
    flowNote.innerHTML = `The apparatus is concept 9's and has not changed since: no score matrix, one grid every token writes into and every query reads from. <strong>Both of this card's switches are scalars sitting on the arrow between one grid and the next.</strong> Under <em>${
      r.label
    }</em> the state is multiplied by ${
      r.gate ? `<code>α</code>` : `<code>1</code> — nothing decays`
    } before ${
      r.delta
        ? "the write, and the write is the difference between the value and what the decayed state already returns for this key"
        : "the value is added on top of it, with nothing reading the state first"
    }. ${
      perToken
        ? `Right now both are read off each token by a declared stand-in projection, so they differ at every step — <strong>${distinct} distinct decay values across ${T} tokens</strong>. <span class="warn">The weights are untrained, so which words open or close the gate is noise; the shape of the effect is not.</span>`
        : "Right now both are constants you set, which is the honest default: this model is untrained, so a projection of its keys carries no meaning a slider does not."
    }`;

    famRead.update({
      norm: fmt(frob, 3),
      err: fmt(err, 4) + (err < 0.5 ? " — reads back well" : ""),
      gates: r.gate ? `${distinct} across ${T} tokens` : "none — this rule has no gate",
      word: `${run.top[0].word} ${fmt(run.top[0].p, 4)}`,
    });
    famNote.innerHTML = `Read the two lines above downward across the four rules and the whole family falls out of one expression. The first line is a small problem solved once per token — <em>stay near where you were, and move towards this pair</em> — and each rule changes exactly one term of it. <strong>α moves what the state is pulled back towards</strong>: not itself, but a faded copy of itself. <strong>β changes what it is pulled towards</strong>: not the value, but the part of the value the state does not already return. This card is the row with both. ${
      r.gate && r.delta
        ? "Both switches are live. Set α to 1 and this row becomes the DeltaNet row exactly; set β to 0 and nothing is ever written at all — the panel below the next one measures both."
        : r.gate
        ? "This row has no delta term, so the write strength slider is inert here: the value is added without anything reading the state first."
        : r.delta
        ? "This row has no gate, so the decay slider is inert here: nothing fades, and what is written stays until something overwrites its direction."
        : "This is the row everything else in this segment is a modification of: add, and never look."
    } The error figure is this state read back with its own ${T} keys — <span class="warn">it is not a quality score</span>: the model is untrained, and ${T} pairs in an ${DH}×${DH} state is already ${
      T > DH ? "past" : "near"
    } the ${DH} independent directions it has to store them in, so every rule here is reading back through some collision. It is here so the effect of the sliders on a real state is a number and not an impression.`;

    // --- 2. the lab
    const prof = labProfile(rule, alpha, beta);
    const ref = labProfile(rule, 1, beta);
    const mean = (a, from, to) => {
      let s = 0;
      for (let i = from; i < to; i++) s += a[i];
      return s / (to - from);
    };
    profile.update({
      points: Array.from(prof, (y, i) => [i, y]),
      reference: Array.from(ref, (y, i) => [i, y]),
      xRange: [0, LAB_N - 1],
      yRange: [-0.1, 1],
      mark: LAB_N - 1,
      markLabel: "newest",
    });
    labBars.update({
      first: { value: Math.max(0, prof[0]), of: 1, text: fmt(prof[0], 3) },
      last: { value: Math.max(0, prof[LAB_N - 1]), of: 1, text: fmt(prof[LAB_N - 1], 3) },
      recent: { value: Math.max(0, mean(prof, LAB_N - 4, LAB_N)), of: 1, text: fmt(mean(prof, LAB_N - 4, LAB_N), 3) },
      all: { value: Math.max(0, mean(prof, 0, LAB_N)), of: 1, text: fmt(mean(prof, 0, LAB_N), 3) },
    });
    const hl = r.gate ? halfLife(alpha) : Infinity;
    const allNow = mean(prof, 0, LAB_N);
    const allRef = mean(ref, 0, LAB_N);
    labRead.update({
      hl: !Number.isFinite(hl) ? "∞ — nothing decays" : `${fmt(hl, 1)} writes`,
      cross: `${fmt(prof[LAB_N - 1] - ref[LAB_N - 1], 3)} on the newest, ${fmt(allNow - allRef, 3)} on the average`,
    });
    labNote.className = "note " + (r.gate && r.delta ? "good" : "");
    labNote.innerHTML = `${LAB_N} random key/value pairs written one after another into a state the size of one of this model's heads (${DH}×${DH}), then the finished state asked for every key in turn — averaged over ${LAB_SEEDS} seeds. The dashed line is the same rule with the decay switched off. <strong>The curve is the point: a gate does not filter, it tilts.</strong> α multiplies the whole state, so it cannot spare the one pair that mattered — everything old fades together and everything recent stands out, which is the two directions of the quoted table below arriving as one shape. ${
      r.delta
        ? `Notice where this rule's curve ends: <strong>the newest pair reads back at ${fmt(
            prof[LAB_N - 1],
            3
          )}</strong>${
            beta === 1
              ? " — exactly 1, at every decay setting. That is not a coincidence and not a fit: at full write strength with unit keys the update forces the state to return the current value for the current key, so whatever the gate just faded, the newest pair is stored perfectly."
              : `, and at full write strength it would be exactly 1 whatever α is. Below full strength the write is partial and the newest pair shares the state with what it did not finish overwriting — which buys the average: ${fmt(
                  allNow,
                  3
                )} across all 64.`
          }`
        : `Without a delta term the newest pair is merely the least-decayed one, not an exact store: ${fmt(
            prof[LAB_N - 1],
            3
          )} here against 1.000 for the rules that read the state before writing. <strong>This is where the gate does the most work in the whole card</strong> — with no decay this rule reads back every pair at about the same mediocre level, and turning the decay up is what makes anything readable at all.`
    }`;

    // --- 3. the corners
    const outOf = (mech) => forward(tokens, { mixer: mech }).trace[0].heads[0];
    const g1 = outOf(ruleMixer("gated", 1, beta));
    const d1 = outOf(ruleMixer("delta", 1, beta));
    let diff = 0;
    for (let i = 0; i < T; i++) for (let a = 0; a < DH; a++) diff = Math.max(diff, Math.abs(g1.out[i][a] - d1.out[i][a]));
    const zeroState = outOf(ruleMixer("gated", alpha, 0)).state;
    let zeroBig = 0;
    for (const row of zeroState) for (const x of row) zeroBig = Math.max(zeroBig, Math.abs(x));
    const full = outOf(ruleMixer("gated", alpha, 1));
    const kLast = featureMap(full.K[T - 1]);
    let exact = 0;
    for (let a = 0; a < DH; a++) exact = Math.max(exact, Math.abs(dot(full.state[a], kLast) - full.V[T - 1][a]));
    cornerRead.update({
      one: diff === 0 ? "0 — bit-identical" : diff.toExponential(2),
      zero: zeroBig === 0 ? "0 — nothing was ever written" : zeroBig.toExponential(2),
      exact: `off by ${exact.toExponential(2)} from the value written with it`,
      mamba: "no — it needs the add rule, which is why the seam now lets that rule decay too",
    });
    cornerNote.innerHTML = `Three claims this card makes, measured on your sentence rather than asserted. <strong>Set the decay to 1 and this mechanism is the previous card's, exactly</strong> — not close, identical to the last bit, which is what "the gate is an addition" has to mean if it means anything. <strong>Set the write strength to 0 and the state is exactly zero</strong>: the decay alone writes nothing, it only fades, so the two switches are genuinely independent rather than two spellings of one. And the third is the one the paper never states: <strong>at full write strength the state returns the value it was just given, to ${exact.toExponential(
      0
    )}</strong>. That follows from the keys being unit vectors — <code>S k = αSk − αSk + v = v</code> — which is exactly what §3.4's L2 normalisation guarantees. The delta rule's precision survives the gate because of a normalisation choice made for a different reason. The fourth line is a limit rather than a result: <code>αS + vkᵀ</code> is not any setting of <code>S(α(I − βkkᵀ)) + βvkᵀ</code>, so the family's four rows are two shapes, not one.`;

    // --- what this page cannot show
    honest.innerHTML = `<strong>The mechanism reproduces here. The benefit does not.</strong> On the lab above, the gate adds almost nothing on top of the delta rule — the last four pairs read back at 0.829 without it and 0.833 at α = 0.95, four thousandths, and the 64-pair average falls the harder it is pushed. That is a real measurement and it is not evidence against the paper; it is evidence about what a browser tab can test. Three reasons it cannot: the keys here are random and near-orthogonal, so there is no correlated junk for a gate to clear, which is exactly the situation §3.2 says gating is for; the model is untrained, so no α is learned and every gate on this card is either a slider you moved or a declared stand-in projection with no meaning behind it; and the paper's numbers are 1.3B parameters over 100B tokens, where the two mechanisms are learned together and the gain shows up as 91.8 against 30.4 on a benchmark this page has no way to run. What this card can honestly show is the rule — that α and β are separable, what each does to a state of a known size, what each costs at both ends of a run — and it leaves the claim of benefit in the quoted panel with its conditions attached. <strong>No throughput number appears on this card either:</strong> the paper's efficiency evidence is Figure 3, a chart with no table, and reading a number off a chart to three digits is the failure this deck exists to avoid.`;
  }

  return { update: render, unmount: () => {} };
}
