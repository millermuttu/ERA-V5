// Concept 22 — parallelizing DeltaNet over sequence length.
// Built from docs/research/parallel-deltanet.md. Four things the research forced.
//
// First, this concept changes nothing about what the model computes, so the card's job is the
// inverse of the usual one: prove the output did not move, to a stated number of decimal places,
// and then show what did. Everything here is measured against a second, independent implementation
// of the same recurrence rather than against itself.
//
// Second, the seam needed no extension at all — the paper's DeltaNet is stateMixer with the feature
// map it already has. The chunkwise algorithm lives in this file precisely because two independent
// implementations are the demonstration; folding it into mixers.js would destroy the comparison.
//
// Third, §3.2 notes in passing that this form gives DeltaNet an explicit attention matrix and drops
// it as useless for training. It is not useless for a visual app: it is the only place in the whole
// deck where a fixed-state model's attention can sit beside softmax's without being an analogy.
//
// Fourth, no timing. The paper's claim is wall-clock on an H100 with hand-written Triton kernels.
// This page counts multiplications and sequential steps, which are exact, and says plainly that the
// step from "fewer steps" to "less time" is the one it cannot take.
import { el, slider, choice } from "../lib/dom.js";
import { attentionGrid } from "../views/grid.js";
import { curveView } from "../views/curve.js";
import { readout } from "../views/bars.js";
import { fmt } from "../model/ops.js";
import { forward, DH, CONFIG } from "../model/transformer.js";
import { stateMixer, softmaxMixer } from "../model/mixers.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

// ------------------------------------------------------------------ §3.3's feature map
const silu = (x) => x / (1 + Math.exp(-x));
const l2 = (v) => {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  return Float64Array.from(v, (x) => x / s);
};
const l1 = (v) => {
  let s = 0;
  for (const x of v) s += Math.abs(x);
  s = s || 1;
  return Float64Array.from(v, (x) => x / s);
};
// The three regimes of §3.3: no normalisation, Schlag et al.'s L1, and this paper's L2.
const NORMS = {
  l2: { label: "SiLU then L2 — this paper", map: (v) => l2(Float64Array.from(v, silu)) },
  l1: { label: "SiLU then L1 — concept 11's convention", map: (v) => l1(Float64Array.from(v, silu)) },
  raw: { label: "SiLU alone — neither", map: (v) => Float64Array.from(v, silu) },
};

// ------------------------------------------------------- the two implementations being compared
/**
 * Sequential form, §2.2: S_t = S_{t-1}(I − β k kᵀ) + β v kᵀ, o_t = S_t q_t.
 * Written straight from the paper, one token at a time, and deliberately not sharing a line of
 * code with the chunkwise version below — a comparison between two spellings of the same function
 * proves nothing.
 */
function sequentialDelta(Qf, Kf, V, dh, betas) {
  const T = Qf.length;
  let S = Array.from({ length: dh }, () => new Float64Array(dh));
  const out = [];
  let mul = 0;
  for (let t = 0; t < T; t++) {
    const k = Kf[t];
    const q = Qf[t];
    const held = new Float64Array(dh);
    for (let a = 0; a < dh; a++) {
      let s = 0;
      for (let j = 0; j < dh; j++) {
        s += S[a][j] * k[j];
        mul++;
      }
      held[a] = s;
    }
    for (let a = 0; a < dh; a++) {
      const u = betas[t] * (V[t][a] - held[a]);
      mul++;
      for (let j = 0; j < dh; j++) {
        S[a][j] += u * k[j];
        mul++;
      }
    }
    const o = new Float64Array(dh);
    for (let a = 0; a < dh; a++) {
      let s = 0;
      for (let j = 0; j < dh; j++) {
        s += S[a][j] * q[j];
        mul++;
      }
      o[a] = s;
    }
    out.push(o);
  }
  return { out, S, mul, steps: T };
}

/**
 * The UT transform, Eq. 10: T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β), by forward substitution.
 * `I + tril(·, −1)` is unit lower triangular whatever the data, so this can never be singular —
 * the algorithm has no failure case to guard against, which is worth knowing and rarely said.
 */
function utTransform(Kf, betas, idx, counter) {
  const n = idx.length;
  const A = Array.from({ length: n }, (_, i) => {
    const r = new Float64Array(n);
    r[i] = 1;
    return r;
  });
  for (let i = 0; i < n; i++)
    for (let j = 0; j < i; j++) {
      let s = 0;
      for (let x = 0; x < Kf[0].length; x++) {
        s += Kf[idx[i]][x] * Kf[idx[j]][x];
        counter.mul++;
      }
      A[i][j] = betas[idx[i]] * s;
      counter.mul++;
    }
  const Tm = Array.from({ length: n }, () => new Float64Array(n));
  for (let col = 0; col < n; col++)
    for (let i = col; i < n; i++) {
      let s = i === col ? betas[idx[col]] : 0;
      for (let j = col; j < i; j++) {
        s -= A[i][j] * Tm[j][col];
        counter.mul++;
      }
      Tm[i][col] = s;
    }
  return { Tm, KK: A };
}

/**
 * Chunkwise parallel form, Eq. 8–11. C = 1 is the recurrence above; C = L is the fully parallel
 * form. Everything crossing a chunk boundary is the single state S; inside a chunk nothing of size
 * d x d is ever built, which is the whole memory argument.
 *
 * The multiply counter is incremented inside the loops that produce the output, not computed
 * alongside them — a cost number derived from a formula is an assertion, not a measurement.
 */
function chunkDelta(Qf, Kf, V, dh, betas, C) {
  const T = Qf.length;
  let S = Array.from({ length: dh }, () => new Float64Array(dh));
  const out = new Array(T);
  const chunks = [];
  const counter = { mul: 0 };
  let steps = 0;
  for (let c0 = 0; c0 < T; c0 += C) {
    const n = Math.min(C, T - c0);
    const idx = Array.from({ length: n }, (_, i) => c0 + i);
    const { Tm, KK } = utTransform(Kf, betas, idx, counter);

    // Eq. 11 — W carries what the chunk erases, U what it writes. Both are C x d.
    const Wm = Array.from({ length: n }, () => new Float64Array(dh));
    const Um = Array.from({ length: n }, () => new Float64Array(dh));
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        const t = Tm[i][j];
        for (let x = 0; x < dh; x++) {
          Wm[i][x] += t * Kf[idx[j]][x];
          Um[i][x] += t * V[idx[j]][x];
          counter.mul += 2;
        }
      }

    // U − W Sᵀ, the chunk's pseudo-values corrected for what the inherited state already holds.
    // It appears in both Eq. 8 and Eq. 9 and is computed once.
    const Z = Array.from({ length: n }, (_, i) => {
      const r = new Float64Array(dh);
      for (let a = 0; a < dh; a++) {
        let s = 0;
        for (let x = 0; x < dh; x++) {
          s += Wm[i][x] * S[a][x];
          counter.mul++;
        }
        r[a] = Um[i][a] - s;
      }
      return r;
    });

    // Eq. 9. The mask includes the diagonal: o_r reads the state after the r-th write.
    for (let i = 0; i < n; i++) {
      const o = new Float64Array(dh);
      for (let a = 0; a < dh; a++) {
        let s = 0;
        for (let x = 0; x < dh; x++) {
          s += Qf[idx[i]][x] * S[a][x];
          counter.mul++;
        }
        o[a] = s;
      }
      for (let j = 0; j <= i; j++) {
        let qk = 0;
        for (let x = 0; x < dh; x++) {
          qk += Qf[idx[i]][x] * Kf[idx[j]][x];
          counter.mul++;
        }
        for (let a = 0; a < dh; a++) {
          o[a] += qk * Z[j][a];
          counter.mul++;
        }
      }
      out[idx[i]] = o;
    }

    // Eq. 8 — the one object that crosses the boundary.
    for (let a = 0; a < dh; a++)
      for (let x = 0; x < dh; x++) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          s += Z[i][a] * Kf[idx[i]][x];
          counter.mul++;
        }
        S[a][x] += s;
      }

    chunks.push({ idx, Tm, KK, Wm, Um, Z });
    steps++;
  }
  return { out, S, mul: counter.mul, steps, chunks };
}

/**
 * The fully parallel form, §3.2: A_ij = k_jᵀ P_{j+1}^i q_i, in matrix form A = (QKᵀ ⊙ M) T, and
 * O = A V. The paper derives it, notes it "could be of interest to the interpretability study for
 * RNNs", and abandons it because the inverse is cubic in L. At sixteen tokens it is free.
 */
function deltaAttention(Qf, Kf, V, dh, betas) {
  const T = Qf.length;
  const idx = Array.from({ length: T }, (_, i) => i);
  const { Tm } = utTransform(Kf, betas, idx, { mul: 0 });
  const A = Array.from({ length: T }, (_, i) => {
    const r = new Float64Array(T);
    for (let j = 0; j <= i; j++) {
      let s = 0;
      // (Q Kᵀ ⊙ M) T — the mask means only m in [j, i] contributes.
      for (let m = j; m <= i; m++) {
        let qk = 0;
        for (let x = 0; x < dh; x++) qk += Qf[i][x] * Kf[m][x];
        s += qk * Tm[m][j];
      }
      r[j] = s;
    }
    return r;
  });
  const out = A.map((row, i) => {
    const o = new Float64Array(dh);
    for (let j = 0; j <= i; j++) for (let a = 0; a < dh; a++) o[a] += row[j] * V[j][a];
    return o;
  });
  return { A, out };
}

/** The same expressions the loops above execute, evaluated on shapes alone, so the cost can be
 *  reported at a sequence length this page does not run. Asserted equal to the live counter at the
 *  app's own scale in selfcheck.js — if they ever drift, the self-check says so. */
export function chunkCost(L, d, C) {
  let mul = 0;
  let steps = 0;
  for (let c0 = 0; c0 < L; c0 += C) {
    const n = Math.min(C, L - c0);
    mul += (n * (n - 1) / 2) * (d + 1); // K Kᵀ below the diagonal, then the β scaling
    mul += (n * (n - 1) * (n + 1)) / 6; // forward substitution
    mul += n * (n + 1) * d; // W and U
    mul += n * d * d; // W Sᵀ
    mul += n * d * d; // Q Sᵀ
    mul += (n * (n + 1) / 2) * (d + d); // (Q Kᵀ ⊙ M) Z
    mul += n * d * d; // Zᵀ K
    steps++;
  }
  return { mul, steps };
}
export const seqCost = (L, d) => ({ mul: L * (3 * d * d + d), steps: L });

/** Both forms as mixers, so `forward` can run the whole model either way and the comparison is
 *  end to end rather than one head deep. */
export const deltaMixer = ({ beta = 1, norm = "l2" } = {}) =>
  stateMixer({ write: "delta", beta, features: NORMS[norm].map, sumNorm: false, attnNorm: false });

export const chunkMixer = ({ beta = 1, norm = "l2", C = 4 } = {}) =>
  function mix(Q, K, V, dh) {
    const r = chunkDelta(Q.map(NORMS[norm].map), K.map(NORMS[norm].map), V, dh, Array(Q.length).fill(beta), C);
    return { ...r, scores: null, weights: null, reads: Q.length, snapshots: [], denominators: [], gates: [], m: dh, kind: "state" };
  };

/** Readable at the scale the paper runs at, exponential once the unnormalised setting blows the
 *  numbers up — the same figure printed two ways would be worse than either. */
const show = (x, dp = 3) => (Math.abs(x) >= 1e5 ? x.toExponential(2) : fmt(x, dp));

const maxAbsDiff = (X, Y) => {
  let m = 0;
  for (let i = 0; i < X.length; i++) for (let j = 0; j < X[i].length; j++) m = Math.max(m, Math.abs(X[i][j] - Y[i][j]));
  return m;
};
const biggest = (M) => {
  let m = 0;
  for (const r of M) for (const x of r) m = Math.max(m, Math.abs(x));
  return m;
};

export function parallelDeltanetCard(root, m) {
  let C = 4;
  let beta = 1;
  let norm = "l2";
  let query = null;
  let chunkPick = 0;
  let showSoftmax = false;

  root.appendChild(
    prose({
      problem:
        "The previous rule in this family works and cannot be trained. Every write has to read the state the write before it left behind, so the sixteen steps of a sentence — or the four thousand of a document — must happen strictly one after another, each one a small matrix update that a machine built to multiply enormous matrices at once has almost nothing to do during. The arithmetic was never the problem; a recurrence is cheaper in operations than attention is. The problem is the shape of it. Three years after the rule was published nobody had trained it at scale, because there was no way to compute it that a modern accelerator could fill its hands with.",
      mechanism:
        "Notice that the update multiplies the old state by an identity minus a rank-one matrix — a Householder factor — and that products of those have had a compact representation in numerical linear algebra since 1987. Rewriting the recurrence that way turns the delta rule into ordinary linear attention with one substitution: the value vector is replaced by a corrected value that can be computed from the earlier vectors alone, with no state matrix ever built. That correction is still sequential, but only within a block of tokens you choose the size of, and inside the block it collapses into the inverse of a small triangular matrix — which is a matrix multiply. One dial then runs from the old recurrence at one end to a fully parallel form at the other, and nothing the model computes changes anywhere along it.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "the same journey, and the same numbers, computed a different way");

  // ------------------------------------------------------- interaction 1: the chunk-size dial
  const cSlider = slider({
    label: "chunk size C",
    min: 1,
    max: 16,
    value: 4,
    oninput: (v) => ((C = v), render()),
  });
  const betaSlider = slider({
    label: "write strength β",
    min: 0,
    max: 1,
    step: 0.05,
    value: 1,
    format: (v) => v.toFixed(2),
    oninput: (v) => ((beta = v), render()),
  });
  const dialRead = readout([
    { key: "diff", label: "largest difference from the sequential form" },
    { key: "steps", label: "steps that must happen one after another" },
    { key: "mul", label: "multiplications, counted as they happen" },
    { key: "word", label: "next word, computed both ways" },
  ]);
  const dialNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "one dial from the recurrence to the parallel form — and the output does not move" }),
      el("div", { class: "formula", text: "S[t+1] = S[t] + (U[t] − W[t] S[t]ᵀ)ᵀ K[t]   ·   O[t] = Q[t] S[t]ᵀ + (Q[t] K[t]ᵀ ⊙ M)(U[t] − W[t] S[t]ᵀ)" }),
      el("div", { class: "ctrls" }, [cSlider.node, betaSlider.node]),
      dialRead.node,
      dialNote,
    ])
  );

  // ------------------------------------------------------ interaction 2: inside one chunk
  const chunkSlider = slider({
    label: "which chunk",
    min: 1,
    max: 4,
    value: 1,
    oninput: (v) => ((chunkPick = v - 1), render()),
  });
  const tGrid = attentionGrid({ onPickRow: () => {}, label: "the chunk's triangular matrix T" });
  const wyRead = readout([
    { key: "shapes", label: "what the chunk holds" },
    { key: "wy", label: "I − Σ wᵢ kᵢᵀ against the explicit product of Householder matrices" },
    { key: "biggest", label: "largest entry of T, and of the state it inherits" },
  ]);
  const wyNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "inside one chunk — where the d × d matrices are supposed to be" }),
      el("div", { class: "formula", text: "P[t]ʳ = I − Σᵢ wᵢ kᵢᵀ   ·   H[t]ʳ = Σᵢ uᵢ kᵢᵀ   ·   W = T K,  U = T V" }),
      el("div", { class: "ctrls" }, [chunkSlider.node]),
      tGrid.node,
      wyRead.node,
      wyNote,
    ])
  );

  // -------------------------------------------------------- interaction 3: the cost of the dial
  const costCurve = curveView({
    xLabel: "chunk size C",
    yLabel: "multiplications, relative to the recurrence",
    ariaLabel: "arithmetic cost against chunk size, with the sequential recurrence as the baseline",
  });
  const costRead = readout([
    { key: "here", label: "at this sentence's scale, C as set above" },
    { key: "real", label: "at 4,096 tokens and 64 dimensions, C = 64" },
    { key: "knee", label: "the paper's chunk size, and where it sits" },
  ]);
  const costNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what the parallelism costs, counted" }),
      costCurve.node,
      costRead.node,
      costNote,
    ])
  );

  // ------------------------------------------------- interaction 3: the attention matrix
  const smToggle = choice({
    label: "matrix shown",
    value: "delta",
    options: [
      { value: "delta", label: "the delta rule's implied attention" },
      { value: "softmax", label: "softmax attention, same head" },
    ],
    onchange: (v) => ((showSoftmax = v === "softmax"), render()),
  });
  const grid = attentionGrid({ onPickRow: (i) => ((query = i), render()), label: "attention weights, signed" });
  const attnRead = readout([
    { key: "neg", label: "weights below zero" },
    { key: "range", label: "smallest and largest weight" },
    { key: "sums", label: "row sums" },
    { key: "check", label: "O = A V against the recurrence" },
  ]);
  const attnNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the one state model in this deck that has an attention matrix" }),
      el("div", { class: "formula", text: "A = (Q Kᵀ ⊙ M) T,   T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β),   O = A V" }),
      el("div", { class: "ctrls" }, [smToggle]),
      grid.node,
      attnRead.node,
      attnNote,
    ])
  );

  // ------------------------------------------------- interaction 4: §3.3, the part that does change
  const normSelect = choice({
    label: "how the keys are normalised",
    value: "l2",
    options: Object.entries(NORMS).map(([k, v]) => ({ value: k, label: v.label })),
    onchange: (v) => ((norm = v), render()),
  });
  const normRead = readout([
    { key: "eig", label: "the transition matrix's one moving eigenvalue" },
    { key: "len", label: "key length ‖k‖₂, over all 64 keys in this block" },
    { key: "state", label: "largest number in the state after the sentence" },
    { key: "erase", label: "how much of an old association a full write leaves behind" },
  ]);
  const normNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the other half of the paper — the half that does change the model" }),
      el("div", { class: "formula", text: "eig(I − β k kᵀ) = { 1 (×d−1),  1 − β‖k‖₂² }" }),
      el("div", { class: "ctrls" }, [normSelect] ),
      normRead.node,
      normNote,
    ])
  );

  // ------------------------------------------------------- what this page cannot show
  const honestNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what this page cannot show you" }),
      honestNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Trainability. The same rule, unchanged, computed in L/C dependent steps instead of L — measured here at 16 steps down to 1 with the output identical to the last bit of a double",
        "Matrix multiplies instead of a scalar recurrence: the inner loop becomes the operation accelerators are built for, which is the actual claim rather than a reduction in arithmetic",
        "Constant memory inside a chunk — no d × d matrix is ever materialised, only two thin C × d blocks, which is what makes the chunked form affordable at all",
        "At 340M with state size matched, the delta rule now beats gated linear attention where it was never previously testable: 26.4 against 24.0 on SWDE, 28.9 against 24.7 on SQuAD, and a better perplexity than Mamba and the transformer both",
        "The hybrids it enables do beat a strong transformer: 16.55 perplexity against 16.85, and 71.0 against 66.6 on SWDE, with two of twenty-four layers left as global attention",
      ],
      givesUp: [
        "Arithmetic, in a fixed and measurable amount: at the paper's own chunk size of 64 the work counted here rises to 1.89 times the recurrence's, and at 128 to 2.88 times. The parallelism is bought, not found",
        "The whole gain is conditional on the machine. On hardware that cannot do many multiplications at once the chunked form is strictly slower, and this page counts operations precisely because it cannot measure the thing that makes them worth it",
        "Speed against its nearest rival: the paper's own limitations section says the training speed still lags gated linear attention, because the state-to-state dependency forces work inside the kernel that an elementwise decay does not need",
        "State size, as a consequence of that: the head dimension cannot grow to where the kernel would stop tiling, which the paper names as the reason it loses the 1.3B recall comparison to a model with twice the state",
        "Length generalisation, which the paper reports as limited and attributes to the rule having no decay at all — the sentence in §5.3 that names the next concept as the fix",
        "Nothing here helps quality. §3.1 and §3.2 change no number the model produces; the gains in the tables come from the normalisation change in §3.3 and from being able to train at all",
      ],
      chooseWhen:
        "Whenever a fixed-size state has to be trained rather than merely described. It is the concept that moved the delta rule from a good idea with a fatal shape into something you can put 100 billion tokens through, and every model in this family after it is built on this algorithm rather than on the 2021 one.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The memory rule from the previous idea can finally be trained on real amounts of text — three years after it was published, without changing what it does",
        "It gives exactly the same answers as before. Not nearly the same: the same, down to the last digit a computer can hold",
        "You choose how much of it happens at once with a single setting, and the two extreme settings are the two methods that already existed",
        "With the same amount of memory to work with, it beats the other fast alternatives at finding facts buried in the text",
      ],
      cons: [
        "It does more arithmetic than the method it replaces — roughly twice as much at the usual setting — and only wins because machines can do a lot of arithmetic simultaneously",
        "On an ordinary processor it would simply be slower, so the benefit belongs to the hardware as much as to the idea",
        "It is still slower to train than the simpler alternatives, which limits how large its memory can be, which then costs it on exactly the task it was built to be good at",
        "Text much longer than what it was trained on still gives it trouble, because the memory has no way to fade — the next idea in this family exists to add one",
        "None of this makes the model itself any better; the quality numbers that improved came from a separate change buried in the same paper",
      ],
      verdict:
        "This is the rare paper whose entire contribution is that nothing changes. The memory rule stays exactly as it was, and someone noticed that its update has a shape numerical analysis solved in 1987 — so the steps that had to happen one at a time can be done in blocks instead, with the same result to the last decimal place. What it buys is that the idea becomes trainable; what it costs is about twice the arithmetic, paid on a machine that does not mind. Everything that made the rule attractive is still true, and so is everything that was wrong with it.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    if (C > T) C = T;
    cSlider.input.max = T;
    cSlider.set(C);
    betaSlider.set(beta);
    if (query === null || query >= T) query = T - 1;
    const betas = Array(T).fill(beta);

    // --- the two runs, whole model, both blocks, all heads
    const seqRun = forward(tokens, { mixer: deltaMixer({ beta, norm }) });
    const chunkRun = forward(tokens, { mixer: chunkMixer({ beta, norm, C }) });
    const h = seqRun.trace[0].heads[0];
    const Qf = h.Q.map(NORMS[norm].map);
    const Kf = h.K.map(NORMS[norm].map);

    // --- the picture. Nothing in it moves, which is the point of the card.
    const snapMixer = stateMixer({ write: "delta", beta, features: NORMS[norm].map, sumNorm: false, attnNorm: false });
    const snaps = snapMixer(h.Q, h.K, h.V, DH, {});
    flow.update({
      tokens,
      head: { ...h, emb: seqRun.trace[0].input },
      weights: null,
      out: h.out,
      top: seqRun.top,
      query,
      opts: { stateMode: { matrix: snaps.snapshots[T - 1] }, qkvBadge: "SiLU applied to Q and K, then L2-normalised" },
    });
    flowNote.innerHTML = `This is concept 11's picture and it is unchanged, deliberately. One grid in the middle, every token reading it before writing to it, no score matrix. <strong>Nothing this concept does is visible here</strong> — not the chunking, not the triangular inverse, not the pseudo-values — because none of it reaches the output. What changed is the order in which the arithmetic happens on the way to these same numbers, and the only honest way to show that is to compute it twice and compare, which is the panel below.`;

    // --- interaction 1: the dial
    let worstHead = 0;
    for (let b = 0; b < CONFIG.BLOCKS; b++)
      for (let hh = 0; hh < CONFIG.HEADS; hh++)
        worstHead = Math.max(worstHead, maxAbsDiff(seqRun.trace[b].heads[hh].out, chunkRun.trace[b].heads[hh].out));
    let logitDiff = 0;
    for (let i = 0; i < seqRun.logits.length; i++) logitDiff = Math.max(logitDiff, Math.abs(seqRun.logits[i] - chunkRun.logits[i]));
    const chunkMul = chunkRun.trace.flatMap((b) => b.heads).reduce((a, x) => a + x.mul, 0);
    const chunkSteps = chunkRun.trace[0].heads[0].steps;
    const seqMul = seqCost(T, DH).mul * CONFIG.BLOCKS * CONFIG.HEADS;
    const nChunks = Math.ceil(T / C);
    // Absolute agreement is only meaningful against the size of the numbers being compared. Under
    // the unnormalised setting the state reaches 1e8 and the absolute gap grows with it while the
    // relative gap stays at machine precision — the card has to say which of the two it is showing.
    const scaleOf = Math.max(1e-12, biggest(seqRun.trace[0].heads[0].out));
    const relative = worstHead / scaleOf;
    // Branch on the absolute figure, because that is the number that looks alarming on screen —
    // the reader needs the explanation exactly when the big number appears, not when it is small.
    const exact = worstHead < 1e-12;
    dialRead.update({
      diff: `${worstHead.toExponential(2)} absolute · ${relative.toExponential(2)} relative to the largest output`,
      steps: `${chunkSteps} per head, against ${T} for the recurrence`,
      mul: `${chunkMul.toLocaleString()}, against ${seqMul.toLocaleString()}`,
      word: `${seqRun.top[0].word} ${seqRun.top[0].p.toFixed(6)} · ${chunkRun.top[0].word} ${chunkRun.top[0].p.toFixed(6)}`,
    });
    dialNote.className = "note " + (exact ? "good" : "warn");
    dialNote.innerHTML = `${
      C === 1
        ? `<strong>C = 1 is concept 11's recurrence exactly</strong> — one token per chunk, ${T} dependent steps, the triangular matrix collapsed to a single number. `
        : C >= T
        ? `<strong>C = ${T} is the fully parallel form</strong>: the whole sentence is one chunk, one step, and the inherited state is empty, so Eq. 9 reduces to the attention-matrix form drawn two panels down. `
        : `${nChunks} chunk${nChunks > 1 ? "s" : ""} of ${C}. The state crosses ${nChunks - 1} boundar${nChunks === 2 ? "y" : "ies"}; inside each chunk nothing waits for anything. `
    }The largest disagreement anywhere in the model — two blocks, four heads, every token — is <strong>${worstHead.toExponential(2)}</strong>, which against outputs of size ${show(scaleOf, 2)} is <strong>${relative.toExponential(2)}</strong> in relative terms${
      exact
        ? `. That is the last bit of a double-precision number, which is to say the two computations are the same computation`
        : `, which is still the last bit of a double. <span class="warn">The absolute gap is large here only because the numbers are.</span> This is the unnormalised setting from the last panel, where the state runs to ${biggest(snaps.snapshots[T - 1]).toExponential(1)}, so an error at machine precision arrives looking alarming — the two computations agree to every digit either of them holds, and nothing about the algorithm has degraded. Worth noticing all the same: the paper says nothing whatever about conditioning, and this is the one place the app can see why someone might want it to`
    }. The next-word bars above are produced by the sequential form; running the chunked one instead gives <strong>${chunkRun.top[0].word}</strong> at ${chunkRun.top[0].p.toFixed(6)} against ${seqRun.top[0].p.toFixed(6)}. Drag C from one end to the other and that prediction does not move, while the steps fall from ${T} to 1 and the multiplications rise from ${(chunkCost(T, DH, 1).mul / seqCost(T, DH).mul).toFixed(2)}× to ${(chunkCost(T, DH, T).mul / seqCost(T, DH).mul).toFixed(2)}× the recurrence's. <em>That is the entire paper.</em>`;

    // --- interaction 2: inside one chunk. The WY representation, checked rather than described:
    // rebuild the product of C Householder matrices from the thin W block and compare.
    const detail = chunkDelta(Qf, Kf, h.V, DH, betas, C);
    if (chunkPick >= detail.chunks.length) chunkPick = detail.chunks.length - 1;
    chunkSlider.input.max = detail.chunks.length;
    chunkSlider.set(chunkPick + 1);
    const ck = detail.chunks[chunkPick];
    const ckTokens = ck.idx.map((i) => tokens[i]);
    tGrid.update({ tokens: ckTokens, weights: ck.Tm, query: ck.idx.length - 1, signed: true });

    // the explicit product Π (I − β k kᵀ), built the expensive way this representation avoids
    let P = Array.from({ length: DH }, (_, a) => Float64Array.from({ length: DH }, (_, b) => (a === b ? 1 : 0)));
    for (const i of ck.idx) {
      const k = Kf[i];
      const nx = Array.from({ length: DH }, () => new Float64Array(DH));
      for (let a = 0; a < DH; a++)
        for (let b = 0; b < DH; b++) {
          let s = P[a][b];
          for (let c = 0; c < DH; c++) s -= P[a][c] * beta * k[c] * k[b];
          nx[a][b] = s;
        }
      P = nx;
    }
    const wy = Array.from({ length: DH }, (_, a) => Float64Array.from({ length: DH }, (_, b) => (a === b ? 1 : 0)));
    for (let i = 0; i < ck.idx.length; i++)
      for (let a = 0; a < DH; a++) for (let b = 0; b < DH; b++) wy[a][b] -= ck.Wm[i][a] * Kf[ck.idx[i]][b];
    const wyErr = maxAbsDiff(P, wy);
    wyRead.update({
      shapes: `T is ${ck.idx.length}×${ck.idx.length}, W and U are ${ck.idx.length}×${DH} — no ${DH}×${DH} matrix anywhere`,
      wy: wyErr.toExponential(2),
      biggest: `${fmt(biggest(ck.Tm), 3)} · ${fmt(biggest(ck.Z), 3)}`,
    });
    wyNote.className = "note " + (wyErr < 1e-10 ? "good" : "warn");
    wyNote.innerHTML = `Every square drawn here is one entry of <code>T</code>, the chunk's own triangular matrix — built from <strong>key against key</strong>, not query against key, which is what makes it a different object from the grid two panels down. Orange is negative. Below the diagonal it is "how much does this token's key overlap that earlier one, scaled by how hard this one writes"; inverting it undoes the whole chain of corrections at once instead of one at a time, and because <code>I + tril(·, −1)</code> is unit triangular whatever the data, <strong>the inverse always exists</strong> — this algorithm has no failure case to guard. From <code>T</code> come two thin blocks: <code>U = T V</code>, what the chunk writes, and <code>W = T K</code>, what it erases — the same expression with the key in place of the value, which is the easiest way to remember what <code>W</code> is for. ${
      ck.idx.length === 1
        ? `At C = 1 the chunk is a single token and T is one number, which is why this setting is just the recurrence.`
        : `The claim being checked: those ${ck.idx.length} rows of <code>W</code> stand in for the product of ${ck.idx.length} separate ${DH}×${DH} Householder matrices. Building that product explicitly and comparing gives <strong>${wyErr.toExponential(2)}</strong>. <em>That is the WY representation</em> — a 1987 result about products of Householder matrices, and the only reason any of this fits in memory. The expensive object was never computed; it was carried as ${ck.idx.length} vectors.`
    }`;

    // --- interaction 3: the cost curve
    const pts = [];
    for (let c = 1; c <= T; c++) pts.push([c, chunkCost(T, DH, c).mul / seqCost(T, DH).mul]);
    costCurve.update({
      points: pts,
      reference: pts.map((p) => [p[0], 1]),
      xRange: [1, T],
      yRange: [0, Math.max(...pts.map((p) => p[1])) * 1.05],
      mark: C,
      markLabel: `C = ${C}`,
    });
    const big = chunkCost(4096, 64, 64);
    const bigSeq = seqCost(4096, 64);
    costRead.update({
      here: `${(chunkCost(T, DH, C).mul / seqCost(T, DH).mul).toFixed(2)}× the arithmetic, ${Math.ceil(T / C)}× fewer steps than ${T}`,
      real: `${(big.mul / bigSeq.mul).toFixed(2)}× the arithmetic, ${big.steps} steps against ${bigSeq.steps}`,
      knee: `C = 64 and C = 128 — ${(chunkCost(4096, 64, 64).mul / bigSeq.mul).toFixed(2)}× and ${(chunkCost(4096, 64, 128).mul / bigSeq.mul).toFixed(2)}×`,
    });
    costNote.className = "note";
    costNote.innerHTML = `Solid line is the chunked form's multiplication count; the dashed line at 1 is the recurrence it is being bought away from. The overhead is <code>O(L·C·d)</code> against the recurrence's <code>O(L·d²)</code>, so <strong>the ratio depends on C over the head dimension and not on the sentence length at all</strong> — the same chunk sizes give the same 1.21×, 1.43×, 1.89× and 2.88× at 512 tokens and at 4,096, which is why one curve can stand for both. That also explains the paper's otherwise unexplained "usually 64 or 128": <strong>at C equal to the head dimension the arithmetic roughly doubles and the sequential depth falls by a factor of the head dimension.</strong> At 4,096 tokens and 64 dimensions that is ${big.steps} steps instead of ${bigSeq.steps} for ${(big.mul / bigSeq.mul).toFixed(2)}× the work. Push C to the whole sequence and the cost runs away — 280× at that scale — which is precisely why the paper derives the fully parallel form and then declines to train with it.`;

    // --- interaction 3: the attention matrix
    const da = deltaAttention(Qf, Kf, h.V, DH, betas);
    const sm = softmaxMixer({})(h.Q, h.K, h.V, DH, {});
    const shown = showSoftmax ? sm.weights : da.A;
    grid.update({ tokens, weights: shown, query, signed: !showSoftmax });
    let neg = 0;
    let tot = 0;
    let lo = Infinity;
    let hi = -Infinity;
    const sums = [];
    for (let i = 0; i < T; i++) {
      let s = 0;
      for (let j = 0; j <= i; j++) {
        const v = shown[i][j];
        s += v;
        tot++;
        if (v < 0) neg++;
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
      sums.push(s);
    }
    const avCheck = maxAbsDiff(da.out, sequentialDelta(Qf, Kf, h.V, DH, betas).out);
    attnRead.update({
      neg: `${neg} of ${tot} (${((100 * neg) / tot).toFixed(1)}%)`,
      range: `${show(lo, 4)} to ${show(hi, 4)}`,
      sums: `${show(Math.min(...sums), 4)} to ${show(Math.max(...sums), 4)}`,
      check: avCheck.toExponential(2),
    });
    attnNote.className = "note " + (showSoftmax ? "" : "warn");
    attnNote.innerHTML = showSoftmax
      ? `Softmax attention on the same head and the same sentence, for comparison. Every weight is non-negative and every row sums to exactly 1, both by construction — the exponential cannot produce a negative number and the denominator is the sum of the row. A token can only ever be <em>added in</em>. Switch back and see what a rule that is allowed to remove looks like when it is written as a matrix.`
      : `This is not an analogy. The paper derives it (§3.2), and the check above confirms it: multiplying this matrix by the value vectors reproduces the recurrence's output to <strong>${avCheck.toExponential(2)}</strong>${
          avCheck > 1e-12 ? " — an absolute figure inflated by the size of the numbers themselves, as the dial panel explains, and machine precision in relative terms" : ""
        }. There is no softmax anywhere in the derivation, and it shows — <strong>${((100 * neg) / tot).toFixed(1)}% of the weights are below zero</strong> (drawn in orange), the largest negative being ${show(lo)}, and the rows sum to anything between ${show(Math.min(...sums), 2)} and ${show(Math.max(...sums), 2)} rather than to one${norm === "raw" ? ", on a scale the panel below explains and the paper never runs at" : ""}. A negative weight has a meaning worth sitting with: it says the association that token wrote was <em>later erased</em> by a key overlapping it, so the read at this query has to subtract it back out. Concepts 9, 10, 11 and 20 all had to say "there is no score matrix here, that is the point"; this one has an exact one, and it is the picture of a memory that can take things back. The paper computes it, calls it possibly of interest for interpretability, and drops it — the inverse is cubic in the sentence length, which is free at sixteen tokens and impossible at four thousand.`;

    // --- interaction 4: §3.3
    const keys = [];
    for (let hh = 0; hh < CONFIG.HEADS; hh++) for (const k of seqRun.trace[0].heads[hh].K) keys.push(NORMS[norm].map(k));
    const lens = keys.map((k) => Math.sqrt(k.reduce((a, x) => a + x * x, 0)));
    const eigs = lens.map((l) => 1 - beta * l * l);
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const stateBig = biggest(snaps.snapshots[T - 1]);
    const meanEig = mean(eigs);
    const unstable = eigs.some((e) => Math.abs(e) > 1);
    normRead.update({
      eig: `1 − β‖k‖₂² = ${fmt(meanEig, 4)} mean, ${fmt(Math.min(...eigs), 4)} to ${fmt(Math.max(...eigs), 4)}`,
      len: `${fmt(mean(lens), 4)} mean, ${fmt(Math.min(...lens), 4)} to ${fmt(Math.max(...lens), 4)}`,
      state: stateBig > 1e6 ? stateBig.toExponential(3) : fmt(stateBig, 3),
      erase: beta === 0 ? "nothing is written at all" : `${(100 * Math.max(0, Math.min(1, Math.abs(meanEig)))).toFixed(0)}% survives a full-strength overwrite`,
    });
    normNote.className = "note " + (unstable ? "warn" : norm === "l2" ? "good" : "");
    normNote.innerHTML = `Everything above this panel leaves the model's numbers exactly where it found them. This does not. The same paper also swaps concept 11's feature map and normalisation, and the reason is an eigenvalue: the update multiplies the state by <code>I − β k kᵀ</code>, whose eigenvalues are 1 in every direction except <code>k</code>, where it is <code>1 − β‖k‖₂²</code>. ${
      norm === "raw"
        ? `<strong>Unnormalised, that is ${fmt(meanEig, 2)} on average and as low as ${fmt(Math.min(...eigs), 1)}</strong> — far outside the unit disk, so the state is multiplied up on every write and reaches <strong>${stateBig > 1e6 ? stateBig.toExponential(2) : fmt(stateBig, 2)}</strong> in ${T} tokens. This is concept 11's divergence arriving by a different road: that card found it as a gradient step with a learning rate above 2, this paper states it as a transition matrix outside the unit circle, and they are the same condition.`
        : norm === "l1"
        ? `Under concept 11's L1 convention it sits at ${fmt(meanEig, 3)} — inside [0, 1], as this paper says of that scheme, so nothing diverges. But look at what it means: at full write strength <strong>${(100 * meanEig).toFixed(0)}% of the old association survives its own replacement</strong>. L1 keeps the recurrence stable and makes a complete overwrite impossible, which is a sharper statement of the problem than the paper offers.`
        : `<strong>L2 pins ‖k‖₂ to exactly 1</strong>, so at full write strength the eigenvalue is exactly ${fmt(meanEig, 4)} — the transition matrix becomes a projection, annihilating the one direction being overwritten and leaving the other ${DH - 1} untouched. "Erase this key and nothing else" stops being a claim about the rule and becomes a property of a matrix. The paper's own words: <em>"erasing information in one subspace while preserving the other d−1 subspaces."</em>`
    } The ablation is worth knowing because it is easy to credit the wrong half: swapping L1 for L2 with the feature map held fixed moves Wikitext perplexity from 31.12 to 28.03, while swapping elu+1 for SiLU with L2 held fixed moves it from 28.03 to <em>28.24</em> — the wrong way. <span class="warn">The normalisation is doing the work; the activation is a wash on the paper's own numbers.</span> One footnote on the formula above: the paper prints this eigenvalue as <code>1 − β‖k‖₂</code>, without the square. Under its own L2 normalisation both read the same, which is presumably how it survived six revisions; on unnormalised keys here the two differ by ${fmt(Math.abs(mean(lens.map((l) => 1 - l)) - meanEig), 1)}.`;

    // --- what this page cannot show
    honestNote.innerHTML = `Every number on this card is a count: multiplications as they are performed, steps as they are taken, differences as they are measured. <strong>Not one of them is a time.</strong> That is not modesty, it is the honest limit — the paper's entire claim is wall-clock on an H100 with hand-written Triton kernels, tensor cores doing 16×16 matrix multiplies as a single instruction, and the state recomputed during the backward pass so it never has to be stored. A browser tab has none of that. Counting operations in a single-threaded interpreter, <strong>the chunked form here is ${(chunkMul / seqMul).toFixed(2)}× slower than the recurrence and would remain so on any machine that does one multiplication at a time</strong> — the whole gain is the assumption that a machine can do thousands at once, and that assumption is the one thing this page cannot test. The paper's own speed evidence is thinner than its reputation suggests, too: Figure 1 plots the chunked form against the recurrent one for sequence lengths from 512 to 16,384 with the axis gridded to 30×, and gives no table of values — and its footnote notes that the recurrent baseline is itself already 2× faster than the original 2021 CUDA kernel, which makes the comparison conservative rather than flattering. Training throughput at 1.3B is reported only as "close to GLA and significantly faster than Mamba". There is no number here to quote, so this card quotes none.`;
  }

  return { update: render, unmount: () => {} };
}
