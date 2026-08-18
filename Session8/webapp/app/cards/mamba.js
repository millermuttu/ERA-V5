// Concept 20 — Selective state space (Mamba).
// Built from docs/research/mamba.md, and the research's first instruction was a boundary: Mamba is
// not an attention variant. It has no Q, no K, no V, no score, no softmax and no head; the block
// replaces the whole attention-plus-MLP sandwich rather than the mixing step inside it. What this
// card is licensed to show is the recurrence and the selection rule, transplanted onto the state
// this app already has — never "Mamba, implemented". Everything else the research settled and this
// card carries: the mechanism is one scalar per token, and Theorem 1 proves the decay and the write
// strength are one decision and its complement, not two knobs; the fixed matrix A stays fixed and
// the transition is still time-varying, because Δ multiplies it; the reason it was affordable is
// concept 14's argument applied to a different object; and Mamba's state is *larger* than what it
// replaces, not smaller, which is the opposite of the folklore.
import { el, slider, choice, toggle } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout, barList } from "../views/bars.js";
import { fmt, mulberry32, gauss, dot } from "../model/ops.js";
import { forward, DH, CONFIG } from "../model/transformer.js";
import { stateMixer } from "../model/mixers.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const N_PAPER = 16; // the paper's default state dimension per channel
const E_PAPER = 2; // its block expansion factor

/** τ_Δ = softplus, the paper's own choice, so Δ is positive whatever the projection returns. */
const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

/** Appendix C with A = −1: Ā = exp(ΔA) and B̄ = 1 − Ā. One decision, and its complement. */
const decayOf = (delta) => Math.exp(-delta);

/** A declared stand-in for the paper's s_Δ(x) = Linear_1(x): the mixer never sees x, only k and v. */
const PROJ = (() => {
  const rnd = mulberry32(4242);
  return Float64Array.from({ length: DH }, () => gauss(rnd));
})();

const halfLife = (g) => (g >= 1 ? Infinity : g <= 0 ? 0 : Math.log(0.5) / Math.log(g));

/**
 * The selective-copying task, at this app's scale and as pure arithmetic.
 *
 * A stream of L positions; m of them carry data, the rest are noise. Each rule ends with the final
 * state being a weighted sum of the tokens it saw, and the weight token t ends with is
 * (1 − g_t) times the product of every decay applied after it. So the task has an exact score with
 * no threshold in it: the m data tokens must be the m largest contributors to the final state.
 */
function copyTrial({ L, m, regular, fixed, seed }) {
  const rnd = mulberry32(seed);
  const isData = new Array(L).fill(false);
  if (regular) {
    // A layout that is a schedule: the data sits at the end, where a single decay can find it.
    for (let i = L - m; i < L; i++) isData[i] = true;
  } else {
    let placed = 0;
    while (placed < m) {
      const p = Math.floor(rnd() * L);
      if (!isData[p]) (isData[p] = true), placed++;
    }
  }

  const weightsFor = (gateOf) => {
    const w = new Array(L).fill(0);
    let tail = 1;
    for (let t = L - 1; t >= 0; t--) {
      const g = gateOf(t);
      w[t] = (1 - g) * tail;
      tail *= g;
    }
    return w;
  };

  // Three rules, in the deck's own vocabulary: never forget, forget on a schedule, forget on command.
  const rules = {
    add: new Array(L).fill(1),
    fixed: weightsFor(() => fixed),
    // The gate is *given* here, not learned: hold the state through noise, take the data in.
    selective: weightsFor((t) => (isData[t] ? 0.5 : 1)),
  };

  const score = (w) => {
    const order = w.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    let hit = 0;
    for (let i = 0; i < m; i++) if (isData[order[i][1]]) hit++;
    return hit / m;
  };
  return { isData, scores: { add: score(rules.add), fixed: score(rules.fixed), selective: score(rules.selective) } };
}

const meanScores = (opts, trials = 30) => {
  const acc = { add: 0, fixed: 0, selective: 0 };
  for (let s = 0; s < trials; s++) {
    const r = copyTrial({ ...opts, seed: 1000 + s });
    for (const k of Object.keys(acc)) acc[k] += r.scores[k];
  }
  for (const k of Object.keys(acc)) acc[k] /= trials;
  return acc;
};

export function mambaCard(root, m) {
  let step = 8;
  let mode = "selective";
  let delta = 1;
  let L = 128;
  let dataCount = 4;
  let regular = false;
  let fixedDecay = 0.9;

  root.appendChild(
    prose({
      problem:
        "Three cards on this timeline already built a fixed-size state, and all three reached it by subtraction from attention — remove the softmax, regroup the sum, then give the write a correction. Every one of them forgets on a schedule set before the data arrives: a plain accumulation never forgets at all, a decayed one fades everything at the same rate whatever the word was, and a segment cache drops whatever falls off the end. None of them can be told, by the text itself, that this token matters and the last forty did not. The same limitation sat in the state-space literature for a different reason: constant dynamics are exactly what let the recurrence be rewritten as a convolution, and the convolution was what made the large state affordable. Efficiency had bought itself a model that cannot select.",
      mechanism:
        "Make the step size a function of the token. One scalar is read off each token, passed through a softplus to keep it positive, and used to discretise a fixed matrix — so the transition applied at each step is different even though the matrix behind it never changes. A large step wipes the state and takes the token; a small one holds the state and ignores the token. The same scalar sets both, because the discretisation forces the write strength to be exactly one minus the decay. Two more projections decide what of the token goes in and what of the state comes out. The cost is that the recurrence can no longer be a convolution, so it has to be run as a scan — and the only reason that is affordable is the insight the timeline already met two cards ago: keep the state in fast memory, never write it out, and recompute it in the backward pass.",
    })
  );

  root.appendChild(
    el("div", { class: "formula", text: "h_t = Ā_t h_{t−1} + B̄_t x_t      Ā_t = exp(Δ_t A)      Δ_t = softplus(Linear(x_t))      with A = −1:  Ā + B̄ = 1" })
  );

  root.appendChild(
    el("div", {
      class: "gate",
      html:
        "<strong>What this card is, and is not.</strong> Mamba is not an attention variant — there is no query, key, value, score, softmax or head anywhere in it, and its block replaces the whole attention-and-MLP sandwich rather than the mixing step inside it. It is on this timeline because it is where <em>the state returns</em>: the same fixed-size memory concepts 9 and 11 built by subtracting from attention, arrived at from the opposite direction and made to work at a scale attention had held alone. So the panels below show <strong>the recurrence and the selection rule, transplanted onto the state this app already has</strong> — a grid whose write is an outer product of a key and a value, which is not the shape of Mamba's state. Read the arrow, not the grid.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root);
  const gateBarsHost = el("div", {});
  const gateNote = el("p", { class: "note" });
  let gateBars = null;
  let gateLen = 0;
  const modePick = choice({
    label: "how it forgets",
    options: [
      { value: "add", label: "never — plain accumulation" },
      { value: "fixed", label: "on a schedule — one constant decay" },
      { value: "selective", label: "on command — a decay per token" },
    ],
    value: mode,
    onchange: (v) => ((mode = v), render()),
  });
  const stepSlider = slider({
    label: "tokens seen",
    min: 1,
    max: 26,
    step: 1,
    value: step,
    format: (v) => `${v}`,
    oninput: (v) => ((step = v), render()),
  });

  // ------------------------------------------------- 1. the gate strip
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the decay, one bar per token" }),
      el("div", { class: "ctrls" }, [modePick, stepSlider.node]),
      gateBarsHost,
      gateNote,
    ])
  );

  // ------------------------------------------------- 2. the selective-copying lab
  const lSlider = slider({
    label: "stream length",
    min: 16,
    max: 256,
    step: 8,
    value: L,
    format: (v) => `${v} positions`,
    oninput: (v) => ((L = v), render()),
  });
  const mSlider = slider({
    label: "tokens to remember",
    min: 2,
    max: 8,
    step: 1,
    value: dataCount,
    format: (v) => `${v}`,
    oninput: (v) => ((dataCount = v), render()),
  });
  const decaySlider = slider({
    label: "the schedule's decay",
    min: 0.5,
    max: 0.999,
    step: 0.001,
    value: fixedDecay,
    format: (v) => v.toFixed(3),
    oninput: (v) => ((fixedDecay = v), render()),
  });
  const spacingToggle = toggle({
    label: "scatter the data at random",
    value: true,
    onchange: (v) => ((regular = !v), render()),
  });
  const labBars = barList({
    rows: [
      { key: "add", label: "never forgets" },
      { key: "fixed", label: "forgets on a schedule", alt: true },
      { key: "selective", label: "forgets on command" },
    ],
  });
  const labRead = readout([
    { key: "reg", label: "on a fixed layout — schedule / command" },
    { key: "rand", label: "scattered at random — schedule / command" },
    { key: "chance", label: "what guessing would score" },
    { key: "paper", label: "quoted: the same task, trained, at length 4096" },
  ]);
  const labNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the task a schedule cannot do" }),
      el("div", { class: "ctrls" }, [lSlider.node, mSlider.node, decaySlider.node, spacingToggle]),
      labBars.node,
      labRead.node,
      labNote,
    ])
  );

  // ------------------------------------------------- 3. one dial, both effects
  const deltaSlider = slider({
    label: "the step size Δ",
    min: -3,
    max: 2,
    step: 0.05,
    value: 0,
    format: (v) => Math.pow(10, v).toPrecision(3),
    oninput: (v) => ((delta = Math.pow(10, v)), render()),
  });
  const deltaCurve = curveView({
    xLabel: "token",
    yLabel: "what one scalar state does",
    ariaLabel: "a scalar state driven by the same input at the chosen step size",
  });
  const deltaRead = readout([
    { key: "keep", label: "Ā — how much of the state survives" },
    { key: "write", label: "B̄ — how much of the token goes in" },
    { key: "sum", label: "their sum" },
    { key: "half", label: "half-life of the past, in tokens" },
  ]);
  const deltaNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "one number, from ignore this token to forget everything else" }),
      el("div", { class: "ctrls" }, [deltaSlider.node]),
      deltaCurve.node,
      deltaRead.node,
      deltaNote,
    ])
  );

  // ------------------------------------------------- 4. on the reader's sentence
  const envCurve = curveView({
    xLabel: "token",
    yLabel: "largest number in the state",
    ariaLabel: "the size of the state as each rule writes into it",
  });
  const envRead = readout([
    { key: "fixedHalf", label: "the schedule's half-life" },
    { key: "selRange", label: "the selective half-life, shortest → longest" },
    { key: "gates", label: "distinct decays applied" },
    { key: "diff", label: "how far the output moved from plain accumulation" },
  ]);
  const envNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "a schedule makes one envelope, a command makes many" }),
      envCurve.node,
      envRead.node,
      envNote,
    ])
  );

  // ------------------------------------------------- 5. the state is not small
  const sizeBars = barList({
    rows: [
      { key: "kv", label: "this app's key/value cache at 16 tokens" },
      { key: "state", label: "this app's state, one block" },
      { key: "mamba", label: "a faithful state at this width", alt: true },
    ],
  });
  const sizeNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the state that returns is bigger than what it replaced" }),
      sizeBars.node,
      sizeNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "The model can be told by the text itself what to forget: a token can reset the state or be skipped entirely, which no schedule fixed before the data arrives can express",
        "On the task built to isolate this, swapping only the layer takes accuracy from 18.3 to 97.0 in the same architecture — gating the architecture instead buys about 38 points and does not finish the task",
        "On induction heads it holds perfect accuracy at a million tokens, four thousand times its training length, where every attention variant tested has already fallen below 60% at one doubling",
        "Inference needs no key/value cache, so the memory per token is constant and far more sequences fit at once — measured as 4–5× the throughput of a similar-sized transformer",
        "It matches a strong modern transformer recipe at every size the paper trained, which no attention-free model had previously done",
      ],
      givesUp: [
        "The recurrence stops being a convolution the moment the parameters vary with the token, so the whole family's efficiency escape hatch is gone and a hand-written scan is required in its place",
        "It is affordable only because of the memory-hierarchy argument from two cards ago — keep the state in fast memory, never write it out, recompute it backwards. Without that, this is a good idea that trains too slowly to test",
        "The state is not small. At this app's width a faithful one is four times the key/value cache it replaces, and training memory is 4% to 12% *more* than an optimised transformer at every batch size the paper measured",
        "There is still no correction: the state decays and the token is added, with nothing reading the state before writing to it, so the only way to revise one association is to fade everything including the ones it had no opinion about",
        "The speed claims are three different multipliers against three different baselines — 20–40× against an unfused scan nobody ships, 7× against the best attention kernel at 32K, 3× against prior state-space implementations — and below about 2,000 tokens attention is simply faster",
        "Nothing was trained above 2.8 billion parameters, which the paper states as its own limitation",
      ],
      chooseWhen:
        "When the workload is a long stream that must be processed once and summarised, and memory per token has to be constant — audio, DNA, high-rate telemetry, or generation where the cache is the bottleneck and no exact recall of a far-back token is required. Not when the task is to find one specific earlier token and read it back verbatim: the state has compressed the past by then, and what it discarded is gone.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The model decides what to remember based on what it is reading, instead of following a fixed rule set before it saw anything",
        "It can skip a stretch of text entirely, or wipe what it was holding and start fresh, and it learns when to do each",
        "Because it keeps a summary rather than a growing list, the memory it needs while running never grows, however long the text",
        "On a task built to test exactly this it goes from barely working to almost perfect, with one change to one part of the model",
      ],
      cons: [
        "The trick that made this family fast in the first place — rewriting the loop as one big filter — stops working, and a specially written routine has to replace it",
        "It is only practical because of a separate insight about where a chip keeps its numbers; without that this could not have been trained at all",
        "The summary it keeps is not small — at training time it uses slightly more memory than a well-tuned ordinary model, not less",
        "It can fade what it holds, but it cannot go back and fix one thing it recorded wrongly without fading everything else too",
      ],
      verdict:
        "The moment the state stops forgetting on a schedule and starts forgetting on command. Everything before it decided how much of the past to keep before any text arrived; this reads one number off each word and lets that number say wipe it all, keep it all, or something between — with the amount taken in forced to be exactly what the amount kept gives up. It is the strongest evidence in this deck that a compressed memory can compete at scale, and it is not the end of the argument: it still cannot correct what it recorded, its own speed numbers point at three different baselines, and the models that followed mostly kept their caches.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    stepSlider.set(Math.min(step, T));
    lSlider.set(L);
    mSlider.set(dataCount);
    decaySlider.set(fixedDecay);

    // Three runs on the same seeded weights and the same sentence — the seam is built for this.
    const selectiveDelta = (i, k) => softplus(dot(k, PROJ));
    const runs = {
      add: forward(tokens, { mixer: stateMixer({ write: "add" }) }),
      fixed: forward(tokens, { mixer: stateMixer({ write: "gated", decay: fixedDecay, beta: 1 - fixedDecay }) }),
      selective: forward(tokens, {
        mixer: stateMixer({
          write: "gated",
          decay: (i, k) => decayOf(selectiveDelta(i, k)),
          beta: (i, g) => 1 - g,
        }),
      }),
    };
    const head = runs[mode].trace[0].heads[0];
    const at = Math.min(step, T) - 1;
    const snap = head.snapshots[at];
    const gates = head.gates.length ? head.gates : new Array(T).fill(1);

    flow.update({
      tokens,
      head: { ...head, emb: runs[mode].trace[0].input },
      weights: null,
      out: head.out,
      top: runs[mode].top,
      query: at,
      opts: {
        stateMode: { matrix: snap },
        qkvBadge: mode === "selective" ? "Δ is read off each token; A is not" : "one decay, fixed before the data",
      },
    });
    flowNote.innerHTML = `The picture is concept 9's, unchanged: no score matrix, one grid that every token writes into and every query reads from. <strong>What this concept changes is the arrow, not the shape</strong> — how much of the grid survives from one token to the next. ${
      mode === "add"
        ? "Right now nothing decays at all: every token is added and stays."
        : mode === "fixed"
        ? `Right now one constant, ${fmt(
            fixedDecay,
            3
          )}, is applied at every step regardless of the word — the schedule fixed before the data arrived.`
        : "Right now the decay is computed from each token as it arrives, so the same grid fades at a different rate at every step."
    } And the grid is where the honesty line falls: this state is a key-dimension by value-dimension matrix, written by an outer product. A faithful state here would be ${N_PAPER} numbers per channel with no key and no value in it, and the write would be a vector times a scalar. The recurrence is the same shape; the object is not.`;

    // --- 1. the gate strip
    if (gateLen !== T) {
      gateLen = T;
      gateBars = barList({ rows: tokens.map((t, i) => ({ key: `g${i}`, label: `${i}  ${t.word}` })) });
      gateBarsHost.replaceChildren(gateBars.node);
    } else {
      gateBarsHost.querySelectorAll(".bar-name").forEach((n, i) => (n.textContent = `${i}  ${tokens[i].word}`));
    }
    const gateVals = {};
    for (let i = 0; i < T; i++) {
      const g = gates[i];
      gateVals[`g${i}`] = {
        value: g,
        of: 1,
        text: `keeps ${fmt(g, 3)} · half-life ${g >= 1 ? "∞" : fmt(halfLife(g), 1)}`,
        tone: i > at ? "alt" : "",
      };
    }
    gateBars.update(gateVals);
    const distinct = new Set(gates.slice(0, T).map((g) => g.toFixed(6))).size;
    gateNote.className = "note " + (mode === "selective" ? "good" : "");
    gateNote.innerHTML = `Each bar is how much of the state survives that token. ${
      mode === "selective"
        ? `<strong>${distinct} distinct values across ${T} tokens</strong> — this is the parameter acquiring a length dimension, which is the entire technical difference between this concept and its predecessor. A parameter that is different at every token is a kernel that is different at every token, and a kernel that is different at every token is not a convolution. That sentence is why the rest of the paper is about writing a scan by hand.`
        : `<strong>One value, repeated ${T} times.</strong> That flat row is what “time-invariant” means, and it is what lets the recurrence be rewritten as a single convolution and run with an FFT. The efficiency of every structured state-space model before this one depended on this row being flat. Switch to the third setting and watch it stop being flat.`
    } The scalar behind each bar is a projection of the token, softplus'd to stay positive — here a declared stand-in, since this app's mixer is handed keys and values rather than the residual stream. <span class="warn">The weights are untrained, so <em>which</em> words open the gate is noise.</span> The shape is not: one rule gives one envelope, the other gives many.`;

    // --- 2. the copying lab
    const now = meanScores({ L, m: dataCount, regular, fixed: fixedDecay });
    const reg = meanScores({ L, m: dataCount, regular: true, fixed: fixedDecay });
    const rand = meanScores({ L, m: dataCount, regular: false, fixed: fixedDecay });
    labBars.update({
      add: { value: now.add, of: 1, text: `${fmt(100 * now.add, 1)}%` },
      fixed: { value: now.fixed, of: 1, text: `${fmt(100 * now.fixed, 1)}%` },
      selective: { value: now.selective, of: 1, text: `${fmt(100 * now.selective, 1)}%` },
    });
    labRead.update({
      reg: `${fmt(100 * reg.fixed, 0)}% / ${fmt(100 * reg.selective, 0)}%`,
      rand: `${fmt(100 * rand.fixed, 0)}% / ${fmt(100 * rand.selective, 0)}%`,
      chance: `${fmt((100 * dataCount) / L, 1)}%`,
      paper: "18.3% → 97.0%",
    });
    labNote.className = "note " + (rand.selective - rand.fixed > 0.3 ? "good" : "warn");
    labNote.innerHTML = `The task is the paper's, at this app's scale and as pure arithmetic. A stream of ${L} positions carries ${dataCount} tokens worth remembering; the rest is noise. Each rule ends with a state that is a weighted sum of what it saw, so the score needs no threshold: <strong>the tokens worth remembering have to be the ${dataCount} largest contributors to the final state.</strong> Guessing scores ${fmt(
      (100 * dataCount) / L,
      1
    )}%. Now use the switch, because it is the whole argument. <strong>On a fixed layout the schedule scores ${fmt(
      100 * reg.fixed,
      0
    )}%</strong> — a single decay is enough, because where the data sits is itself a schedule and the decay can be tuned to it. <strong>Scatter the same data at random and it falls to ${fmt(
      100 * rand.fixed,
      0
    )}%</strong>, while forgetting on command stays at ${fmt(
      100 * rand.selective,
      0
    )}%. Nothing about the model changed between those two numbers — one switch changed where the data sits. That is the paper's Figure 2: <em>“the spacing between inputs-to-outputs is varying and cannot be modeled by static convolution kernels.”</em> <span class="warn">The gate here is given, not learned</span> — it is told which tokens are data. The paper's own version of this row is trained, at length 4,096, and moves 18.3% to 97.0% by swapping only the layer inside the same architecture.`;

    // --- 3. Δ
    const keep = decayOf(delta);
    const write = 1 - keep;
    const trace = [];
    let h = 0;
    const drive = (t) => (t % 4 === 0 ? 1 : 0.1);
    for (let t = 0; t < 32; t++) {
      h = keep * h + write * drive(t);
      trace.push([t, h]);
    }
    deltaCurve.update({
      points: trace,
      reference: Array.from({ length: 32 }, (_, t) => [t, drive(t)]),
      xRange: [0, 31],
      yRange: [0, 1.1],
    });
    deltaRead.update({
      keep: fmt(keep, 4),
      write: fmt(write, 4),
      sum: fmt(keep + write, 4),
      half: keep >= 0.9999 ? "∞" : fmt(halfLife(keep), 2),
    });
    deltaNote.className = "note";
    deltaNote.innerHTML = `Pure arithmetic — no model runs here. The dashed line is a fixed input, spiking every fourth token; the solid line is one scalar state driven by it at the step size you have chosen. At small Δ the state barely moves: the token is <em>ignored</em>. At large Δ the state is the token: everything before it has been <em>wiped</em>. The paper's own sentence is <em>“a large Δ resets the state h and focuses on the current input x, while a small Δ persists the state and ignores the current input.”</em> <strong>Watch the third readout while you drag.</strong> It never leaves ${fmt(
      keep + write,
      4
    )} — how much of the past survives and how much of the token goes in are not two settings, they are one decision and its complement, which the paper proves in four lines for the case where the matrix is −1. Note what is <em>not</em> input-dependent: that matrix. It stays a fixed learned parameter, and the transition is still different at every token, because the step size multiplies it inside an exponential. A card saying this concept makes the matrix input-dependent would be wrong; so would one saying its transition is fixed.`;

    // --- 4. the envelopes
    const envOf = (r) =>
      r.trace[0].heads[0].snapshots.map((s, i) => [i, Math.max(...s.flatMap((row) => Array.from(row, Math.abs)))]);
    const envAdd = envOf(runs.add);
    const envFixed = envOf(runs.fixed);
    const envSel = envOf(runs.selective);
    const top = Math.max(...envAdd.map((p) => p[1]), ...envFixed.map((p) => p[1]), ...envSel.map((p) => p[1]));
    envCurve.update({
      points: envSel,
      reference: envFixed,
      xRange: [0, T - 1],
      yRange: [0, top * 1.1],
    });
    envCurve.setDots(
      envAdd.map(([x, y]) => [x, y, "ref"]),
      [0, T - 1],
      [0, top * 1.1]
    );
    const selGates = runs.selective.trace[0].heads[0].gates.slice(0, T);
    const lives = selGates.map(halfLife).filter(Number.isFinite);
    const outDiff = (() => {
      const a = runs.add.trace[0].heads[0].out;
      const b = runs.selective.trace[0].heads[0].out;
      let d = 0;
      for (let i = 0; i < T; i++) for (let x = 0; x < DH; x++) d = Math.max(d, Math.abs(a[i][x] - b[i][x]));
      return d;
    })();
    envRead.update({
      fixedHalf: `${fmt(halfLife(fixedDecay), 1)} tokens`,
      selRange: `${fmt(Math.min(...lives), 2)} → ${fmt(Math.max(...lives), 2)}`,
      gates: `${new Set(selGates.map((g) => g.toFixed(6))).size} of ${T}`,
      diff: fmt(outDiff, 3),
    });
    envNote.className = "note";
    envNote.innerHTML = `Three rules, the same seeded weights, your sentence. The dots are plain accumulation, which only ever grows. The dashed line is one constant decay: a smooth geometric envelope with a single half-life of ${fmt(
      halfLife(fixedDecay),
      1
    )} tokens, the same at every word. The solid line is a decay read off each token, and the shape to notice is that it is <strong>ragged</strong> — the half-life ranges from ${fmt(
      Math.min(...lives),
      2
    )} to ${fmt(
      Math.max(...lives),
      2
    )} tokens across this one sentence, with visible drops where a token asked for the state to be cleared. <strong>One scalar against a distribution is the whole concept in one readout.</strong> And the outputs differ by ${fmt(
      outDiff,
      3
    )} from plain accumulation, which is the point worth being blunt about: this is a different function, not a faster way to compute the same one. <span class="warn">Untrained weights, so read the envelope and not the words</span> — which token opens the gate here is noise; that there is a distribution of gates at all is not.`;

    // --- 5. the size of the state
    const kv = 2 * T * DH;
    const appState = CONFIG.HEADS * (DH * DH + DH);
    const faithful = E_PAPER * CONFIG.D * N_PAPER;
    const ofBar = Math.max(kv, appState, faithful);
    sizeBars.update({
      kv: { value: kv, of: ofBar, text: `${kv} numbers · grows with the text` },
      state: { value: appState, of: ofBar, text: `${appState} numbers · fixed` },
      mamba: { value: faithful, of: ofBar, text: `${faithful} numbers · fixed` },
    });
    sizeNote.className = "note warn";
    sizeNote.innerHTML = `<strong>“The state returns” does not mean “the state gets smaller”.</strong> At this app's width, a faithful state — the paper's ${N_PAPER} numbers per channel, with its block's expansion of ${E_PAPER} — is <strong>${faithful} numbers per block</strong>, about ${fmt(
      faithful / appState,
      1
    )}× this app's own fixed state and ${fmt(
      faithful / kv,
      1
    )}× the key/value cache it replaces at ${T} tokens. That is not an accident of the toy: the whole point of the state-space line is a state <em>larger</em> than a traditional recurrent model's, and the paper's central engineering section exists because that state is expensive to move, not to compute. Its own measured training memory is <strong>4% to 12% more</strong> than an optimised transformer at every batch size it reports — 4.8 GB against 4.6, 38.2 against 34.5. The constant-memory win is real and it is at <em>inference</em>, where there is no key/value cache at all; anyone who has transferred that to training is thinking of the wrong table. And what makes the big state affordable is not this paper's idea: it is the memory-hierarchy argument from two concepts ago, by the same author — keep it in fast memory, never write it out, recompute it going backwards. Nothing in a browser at ${T} tokens can demonstrate that, so this card does not pretend to.`;
  }

  return { update: render, unmount: () => {} };
}
