// Concept 11 — the delta rule, fast weight programmers.
// Built from docs/research/delta-rule.md. Three things the research forced. First, the paper keeps
// two claims apart that most summaries merge: capacity belongs to the feature map (§4.1, §5.1) and
// the delta rule does not raise it — both labs below are here so the card cannot cheat on that.
// Second, the app's delta branch was divergent until the paper's sum normalisation (Eq. 29) went
// in, and the read is no longer divided by the running denominator, which this paper rejects for
// delta models. Third, the paper's `v_new` is the blend that gets stored, not the incoming value,
// so nothing on screen uses the bare word "new".
import { el, slider, toggle, choice } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout } from "../views/bars.js";
import { mulberry32, gauss, fmt } from "../model/ops.js";
import { forward, DH } from "../model/transformer.js";
import { stateMixer } from "../model/mixers.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

// ---------------------------------------------------------------- the labs
// A standalone associative memory, deliberately separate from the sentence: capacity is a claim
// about the number of pairs written, and the sentence is a fixed 16 tokens. The state is DH x d_dot,
// matching the model's per-head state exactly.
//
// Features are L2-normalised rather than sum-normalised here. Sum normalisation (what the running
// model uses, Eq. 29) leaves ‖φ(k)‖² near 0.18 on this data, so one delta write corrects only 18%
// of the error and "exact" becomes "approximately". L2 makes the effective learning rate β‖φ(k)‖²
// exactly β, so the demo is exactly exact. It is a simplification, and the caption says so.
const unit = (v) => {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  return Float64Array.from(v, (x) => x / s);
};
const rvec = (rnd, n = DH) => Float64Array.from({ length: n }, () => gauss(rnd));

/** n vectors, mutually orthogonal while the dimension allows it. Past `dim` the residual collapses
 *  to numerical noise and the next vector is necessarily a combination of the ones before it —
 *  which is §4.1's bound arriving on its own rather than being imposed. */
function orthoVecs(rnd, n, dim = DH) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const v = rvec(rnd, dim);
    for (const u of out) {
      let d = 0;
      for (let a = 0; a < dim; a++) d += v[a] * u[a];
      for (let a = 0; a < dim; a++) v[a] -= d * u[a];
    }
    let s = 0;
    for (const x of v) s += x * x;
    out.push(unit(Math.sqrt(s) < 1e-8 ? rvec(rnd, dim) : v));
  }
  return out;
}

/** Appendix C in four lines: concatenate the positive and negative parts, roll, multiply.
 *  d_dot = 2 · d_key · ν, so the capacity the state can hold moves with ν and nothing else. */
function dpfp(v, nu) {
  const x = [...Array.from(v, (a) => Math.max(0, a)), ...Array.from(v, (a) => Math.max(0, -a))];
  const out = [];
  for (let j = 1; j <= nu; j++) for (let i = 0; i < x.length; i++) out.push(x[i] * x[(i + j) % x.length]);
  return Float64Array.from(out);
}

const elu1 = (x) => (x > 0 ? x + 1 : Math.exp(x));
const PHI = {
  none: { label: "the key itself", dim: DH, map: (k) => Float64Array.from(k) },
  elu: { label: "elu(x)+1", dim: DH, map: (k) => Float64Array.from(k, elu1) },
  1: { label: "DPFP-1", dim: 2 * DH, map: (k) => dpfp(k, 1) },
  2: { label: "DPFP-2", dim: 4 * DH, map: (k) => dpfp(k, 2) },
  3: { label: "DPFP-3", dim: 6 * DH, map: (k) => dpfp(k, 3) },
};
const feature = (k, phi) => unit(PHI[phi].map(k));

/** One pass of writes over a fixed-size state, under either rule. */
function writeAll(feats, vals, rule, beta) {
  const m = feats[0].length;
  const S = Array.from({ length: DH }, () => new Float64Array(m));
  for (let t = 0; t < feats.length; t++) {
    const f = feats[t];
    for (let a = 0; a < DH; a++) {
      let held = 0;
      if (rule === "delta") for (let b = 0; b < m; b++) held += S[a][b] * f[b];
      const target = rule === "add" ? vals[t][a] : beta * (vals[t][a] - held);
      for (let b = 0; b < m; b++) S[a][b] += target * f[b];
    }
  }
  return S;
}

/** Mean relative recovery error: read every key back, compare with what should be bound to it. */
function recoveryError(S, feats, targets) {
  let e = 0;
  for (let j = 0; j < feats.length; j++) {
    let num = 0;
    let den = 0;
    for (let a = 0; a < DH; a++) {
      let r = 0;
      for (let b = 0; b < feats[j].length; b++) r += S[a][b] * feats[j][b];
      num += (r - targets[j][a]) ** 2;
      den += targets[j][a] ** 2;
    }
    e += Math.sqrt(num) / Math.sqrt(den || 1);
  }
  return e / feats.length;
}

/**
 * Setting 1 — n distinct pairs, one write each, read them all back.
 *
 * The two geometries are deliberately different objects. "ortho" builds the feature vectors
 * directly as a mutually orthogonal set: that is the paper's ideal case, the one the capacity
 * bound is stated about, and no feature map actually reaches it — which is the point. "rand" runs
 * real random keys through φ, which is what a model has.
 */
function capacityTrial(n, geom, phi, beta, trials) {
  let a = 0;
  let d = 0;
  for (let s = 0; s < trials; s++) {
    const rnd = mulberry32(1000 + s);
    const feats =
      geom === "ortho"
        ? orthoVecs(rnd, n, PHI[phi].dim)
        : Array.from({ length: n }, () => feature(unit(rvec(rnd)), phi));
    const vals = Array.from({ length: n }, () => rvec(rnd));
    a += recoveryError(writeAll(feats, vals, "add", beta), feats, vals);
    d += recoveryError(writeAll(feats, vals, "delta", beta), feats, vals);
  }
  return [a / trials, d / trials];
}

/** Setting 2 — S unique keys, L writes sampled with replacement, recover the most recent value. */
function overwriteTrial(uniq, writes, beta, trials = 24) {
  let a = 0;
  let d = 0;
  for (let s = 0; s < trials; s++) {
    const rnd = mulberry32(7000 + s);
    const feats = orthoVecs(rnd, uniq);
    const seqK = [];
    const seqV = [];
    const last = new Map();
    for (let t = 0; t < writes; t++) {
      const j = Math.min(uniq - 1, Math.floor(rnd() * uniq));
      const v = rvec(rnd);
      seqK.push(feats[j]);
      seqV.push(v);
      last.set(j, v);
    }
    const targets = feats.map((_, j) => last.get(j) || new Float64Array(DH));
    a += recoveryError(writeAll(seqK, seqV, "add", beta), feats, targets);
    d += recoveryError(writeAll(seqK, seqV, "delta", beta), feats, targets);
  }
  return [a / trials, d / trials];
}

const STAGES = [
  ["w1", "after the first write"],
  ["remove", "remove — subtract what the key already holds"],
  ["write", "write — add the blend"],
  ["done", "after the second write"],
];

export function deltaRuleCard(root, m) {
  let beta = 1;
  let stage = "done";
  let geom = "ortho";
  let phi = "none";
  let uniq = 4;
  let writes = 16;
  let step = null;

  root.appendChild(
    prose({
      problem:
        "The previous concept compressed the whole past into one fixed grid, and every token wrote into it by addition. Addition can only ever say “and also”. If a key is bound to one value and later has to mean something else, nothing in the rule can take the first association out — both outer products stay in the sum, both keep consuming room, and the read returns their superposition. On top of that, a grid of this size can hold only so many associations before the reads interfere with each other at all.",
      mechanism:
        "Before writing, read. Ask the state what it currently returns for the incoming key, call that the held value, and write only the difference between what you wanted to store and what is already there — scaled by a write strength between zero and one that the model produces for each token. The paper's framing is that the state was never really a state: it is a weight matrix that the tokens are programming, and once you see it that way the right update is the oldest rule in the field, a gradient step on the squared error between the value you want and the value the memory gives back.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "the same journey, with the state corrected rather than piled up");

  // -------------------------------------------------- interaction 1: 40 then 55
  const betaSlider = slider({
    label: "write strength β",
    min: 0,
    max: 1,
    step: 0.05,
    value: 1,
    format: (v) => v.toFixed(2),
    oninput: (v) => ((beta = v), render()),
  });
  const stageRow = el("div", { class: "stages" });
  const stageBtns = STAGES.map(([key, label]) =>
    el("button", {
      class: "stage",
      type: "button",
      text: label,
      onclick: () => ((stage = key), render()),
    })
  );
  stageRow.append(...stageBtns);
  const twoRead = readout([
    { key: "vin", label: "v_in, this token's value" },
    { key: "vheld", label: "v_held, what the key returns now" },
    { key: "delta", label: "Δ = v_in − v_held" },
    { key: "back", label: "read back after the write" },
    { key: "add", label: "the add-only rule reads back" },
  ]);
  const twoNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "one key, written twice — 40, then 55" }),
      el("div", { class: "formula", text: "W ← W + β (v_in − W φ(k)) ⊗ φ(k)" }),
      el("div", { class: "ctrls" }, [betaSlider.node]),
      stageRow,
      twoRead.node,
      twoNote,
    ])
  );

  // ------------------------------------------------ interaction 2: the capacity lab
  const geomToggle = toggle({
    label: "keys arranged not to overlap",
    value: true,
    onchange: (v) => ((geom = v ? "ortho" : "rand"), render()),
  });
  const phiSelect = choice({
    label: "feature map φ",
    value: "none",
    options: Object.entries(PHI).map(([k, v]) => ({ value: k, label: `${v.label} — d_dot ${v.dim}` })),
    onchange: (v) => ((phi = v), render()),
  });
  const capCurve = curveView({
    xLabel: "pairs written into the state",
    yLabel: "how wrong the read-back is",
    ariaLabel: "recovery error against the number of stored pairs, for the add rule and the delta rule",
  });
  const capRead = readout([
    { key: "cliff", label: "at capacity, then one past it" },
    { key: "cap", label: "what the state can hold" },
    { key: "gap", label: "delta against add, at the far end" },
  ]);
  const capNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the ceiling — and the fact that this rule does not raise it" }),
      el("div", { class: "ctrls" }, [phiSelect, geomToggle]),
      capCurve.node,
      capRead.node,
      capNote,
    ])
  );

  // ---------------------------------------------- interaction 3: the overwrite lab
  const uniqSlider = slider({
    label: "distinct keys",
    min: 2,
    max: 16,
    value: 4,
    oninput: (v) => ((uniq = v), render()),
  });
  const writeSlider = slider({
    label: "writes, sampled with replacement",
    min: 2,
    max: 40,
    value: 16,
    oninput: (v) => ((writes = v), render()),
  });
  const owRead = readout([
    { key: "delta", label: "delta rule, recovering the latest value" },
    { key: "add", label: "add-only rule, same sequence" },
    { key: "regime", label: "distinct keys against capacity" },
  ]);
  const owNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the same key, written over and over — where this rule wins outright" }),
      el("div", { class: "ctrls" }, [uniqSlider.node, writeSlider.node]),
      owRead.node,
      owNote,
    ])
  );

  // --------------------------------------------- what the running model does with it
  const stepSlider = slider({
    label: "tokens seen",
    min: 1,
    max: 16,
    value: 16,
    oninput: (v) => ((step = v), render()),
  });
  const liveRead = readout([
    { key: "delta", label: "largest number in the state, delta rule" },
    { key: "add", label: "same, add-only" },
    { key: "raw", label: "delta without the paper's normalisation" },
  ]);
  const liveNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "why the paper spends two pages on a division" }),
      el("div", { class: "ctrls" }, [stepSlider.node]),
      liveRead.node,
      liveNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "An association can be replaced rather than merely added to: with the distinct keys inside capacity, the most recent value comes back exactly, for any number of overwrites",
        "It closes roughly two-thirds of the gap between the linear model and a full Transformer on WikiText-103 — 38.3 test perplexity for the add rule against 35.5 for this one, with 34.1 the Transformer's",
        "The gain survives outside the overcapacity regime as well — 38.8 against 35.0 in the paper's own ablation, so this is not only a crowding fix",
        "Carried across segments with no context limit at all it scores 29.4 where the add rule scores worse than 260, at identical state size",
        "The cost is one extra read per token: 63,000 against 66,000 words per second, and one gigabyte more memory",
      ],
      givesUp: [
        "Parallel training. Each write must read the state the previous write left, so the sequence can no longer be computed as one batched product — the paper never mentions this, and it is why the rule waited three years to be scaled",
        "Nothing about capacity. The ceiling belongs to the feature map, and both rules fail identically the moment the number of distinct keys passes it — as the first lab shows with zero error at eight pairs and 0.4 at nine",
        "Stability without care: the update is a gradient step whose learning rate is the squared length of the feature vector, and the paper reports its models diverged unless the features were normalised first",
        "It still loses to a full Transformer — 35.5 against 34.1 — and to Transformer-XL when that model is given eight to fifty times the state",
        "The write strength is one number per token for the whole value vector: it cannot overwrite part of an association and keep the rest",
      ],
      chooseWhen:
        "When the past has to live in a fixed-size memory and keys genuinely recur — which in language they always do. It is the correct update rule for a compressed memory, and the two concepts that follow it in this family exist to make it affordable and to let it forget.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The model can change its mind: if a word means something different later, the new meaning replaces the old one instead of being mixed with it",
        "When the number of things being remembered stays within what the memory can hold, the latest answer comes back perfectly, no matter how many times it was rewritten",
        "It costs almost nothing — about one part in twenty of the speed, and the quality gain is large enough to close most of the gap to the far more expensive method",
      ],
      cons: [
        "Each step has to look at the memory before changing it, so the steps can no longer be done all at once — training gets much slower in a way the paper does not mention",
        "It does not make the memory bigger. Past a certain number of distinct things, everything blurs together exactly as before",
        "It has no way to forget anything it is not currently overwriting, so a memory that runs for a long time fills up and stays full",
        "The strength of a correction is a single dial for the whole entry: it cannot half-replace one part and leave another",
      ],
      verdict:
        "Read before you write, and write only the difference. That one change turns a memory that can only pile things up into one that can be corrected, and it is worth most of the quality gap to the expensive method it is imitating. What it does not do is make the memory any larger — and the price is that the steps must now happen one after another, which is the bill the next few concepts in this family come to settle.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    betaSlider.set(beta);
    uniqSlider.set(uniq);
    writeSlider.set(writes);
    if (step === null || step > T) step = T;
    stepSlider.input.max = T;
    stepSlider.set(Math.min(step, T));
    for (let i = 0; i < STAGES.length; i++) stageBtns[i].className = "stage" + (STAGES[i][0] === stage ? " is-on" : "");

    // --- the running model, delta rule
    const runDelta = forward(tokens, { mixer: stateMixer({ write: "delta", beta }) });
    const h = runDelta.trace[0].heads[0];
    const snap = h.snapshots[Math.min(step, T) - 1];
    flow.update({
      tokens,
      head: { ...h, emb: runDelta.trace[0].input },
      weights: null,
      out: h.out,
      top: runDelta.top,
      query: Math.min(step, T) - 1,
      opts: { stateMode: { matrix: snap } },
    });
    flowNote.innerHTML = `The picture is concept 9's, unchanged: no score matrix, one grid in the middle that every token writes into and every query reads from. What changed is inside the arrow going in. Each token now <strong>reads the grid first</strong>, at its own key, and writes only the difference between the value it wanted stored and the value it found. Nothing else in the pipeline moves — same projections, same output path, same prediction bars.`;

    // --- interaction 1: 40 then 55, on one key
    const V1 = 40;
    const V2 = 55;
    const held = V1;
    const blend = beta * V2 + (1 - beta) * held; // Eq. 22
    const shown = { w1: V1, remove: 0, write: blend, done: V1 + beta * (V2 - V1) }[stage];
    twoRead.update({
      vin: stage === "w1" ? fmt(V1, 1) : fmt(V2, 1),
      vheld: stage === "w1" ? "0.0 — the state is empty" : fmt(held, 1),
      delta: stage === "w1" ? fmt(V1, 1) : fmt(V2 - held, 1),
      back: fmt(shown, 2),
      add: stage === "w1" ? fmt(V1, 1) : fmt(V1 + V2, 1),
    });
    twoNote.className = "note " + (beta === 1 ? "good" : "");
    twoNote.innerHTML =
      stage === "w1"
        ? `First write into an empty state. There is nothing held at this key, so the difference is the whole value and both rules agree: the read comes back <strong>40.0</strong>. The rules only ever differ on the second write.`
        : stage === "remove"
        ? `The paper writes the update as two named operations, and this is the first: <em>subtract</em> the outer product of the value the key currently holds. The read at this key is now exactly <strong>0.0</strong> — the old association has been lifted out, which is precisely what the add-only rule has no way of doing.`
        : stage === "write"
        ? `And the second: add back a blend of the incoming value and the held one, ${fmt(beta, 2)} of the new and ${fmt(1 - beta, 2)} of the old. The read is <strong>${fmt(blend, 2)}</strong>. Compressed into one line those two operations become “add β times the difference”, which is the form usually quoted.`
        : `Both writes done. The delta rule reads back <strong>${fmt(V1 + beta * (V2 - V1), 2)}</strong>; the add-only rule reads back <strong>95.0</strong>, because the 40 is still sitting there and nothing in that rule can remove it. Note what the state actually learned at full strength: not 55, but the correction 15 — and 40 plus 15 is 55. Drag the write strength and the read back traverses 40 to 55 continuously, which is what “how much of this should replace what is there” means as a number. In the real model this dial is not a dial: it is one learned number per token, per head, squeezed through a sigmoid.`;

    // --- interaction 2: the capacity lab
    const dot = PHI[phi].dim;
    const maxN = 26;
    const pts = [];
    const ref = [];
    for (let n = 1; n <= maxN; n++) {
      const [a, d] = capacityTrial(n, geom, phi, 1, 10);
      pts.push([n, d]);
      ref.push([n, a]);
    }
    const at = (n) => pts.find((p) => p[0] === n)?.[1] ?? 0;
    const atA = (n) => ref.find((p) => p[0] === n)?.[1] ?? 0;
    const yMax = Math.max(1, ...ref.map((p) => p[1])) * 1.05;
    capCurve.update({
      points: pts,
      reference: ref,
      xRange: [1, maxN],
      yRange: [0, yMax],
      mark: Math.min(dot, maxN),
      markLabel: dot <= maxN ? `capacity ${dot}` : "",
    });
    const onChart = dot < maxN;
    capRead.update({
      cliff: onChart ? `${dot}: ${at(dot).toFixed(3)} · ${dot + 1}: ${at(dot + 1).toFixed(3)}` : "past the end of this chart",
      cap: `${dot} pairs`,
      gap: `${at(maxN).toFixed(3)} vs ${atA(maxN).toFixed(3)}`,
    });
    capNote.className = "note";
    capNote.innerHTML =
      geom === "ortho"
        ? `Solid line is the delta rule, dashed is add-only, and the keys here are arranged so that no two of them overlap at all — the paper's ideal case, the one the bound is stated about. ${
            onChart
              ? `<strong>Both rules are exactly zero up to ${dot} pairs and both break at ${dot + 1}</strong> — ${at(dot).toFixed(3)} then ${at(dot + 1).toFixed(3)}, with nothing changed but the count.`
              : `With ${PHI[phi].label} the ceiling is ${dot} pairs, past the right-hand edge of this chart, so both lines sit at exactly zero across the whole of it — the wall has not gone away, it has moved out of view.`
          } Reading this memory is a matrix multiply, so two associations stay separate only while their keys do not overlap, and a space of ${dot} dimensions holds at most ${dot} non-overlapping directions; the next key is forced to be a mixture of the others. <strong>The delta rule does not move that wall.</strong> The feature map does — change φ above and the marker and the cliff move together, which is the whole difference between §4.1 and §5.1 of the paper. Anyone who says this rule fixes the capacity problem has merged two claims it keeps a section apart.`
        : `Now the honest version: real random keys put through the feature map, rather than an arrangement chosen to be perfect. Nothing is cleanly separated any more, so both lines leave zero at the second pair and climb from there — the ceiling above is an upper bound no feature map actually reaches. Here the delta rule does help, and steadily: ${at(maxN).toFixed(3)} against ${atA(maxN).toFixed(3)} at ${maxN} pairs, roughly a third less wrong throughout. Watch what φ does in this mode: ${PHI[phi].label} spreads the keys through ${dot} dimensions, and elu(x)+1 is the worst of the options at every count precisely because its output is positive in every coordinate — every key ends up in the same corner of the space, so it never comes close to the ${DH} it nominally has room for.`;

    // --- interaction 3: the overwrite lab
    const [owAdd, owDelta] = overwriteTrial(uniq, Math.max(writes, uniq), beta);
    const over = uniq > DH;
    owRead.update({
      delta: owDelta.toFixed(4),
      add: owAdd.toFixed(4),
      regime: `${uniq} against ${DH}${over ? " — over" : ""}`,
    });
    owNote.className = "note " + (over ? "warn" : "good");
    owNote.innerHTML = over
      ? `Past capacity even this rule degrades: ${owDelta.toFixed(3)} against add-only's ${owAdd.toFixed(3)}. Better, but no longer exact — because the keys themselves can no longer be told apart, and correcting an association you cannot address precisely is still a smudge. Pull the distinct keys back to ${DH} or fewer and the error drops to zero.`
      : `${uniq} distinct keys, ${Math.max(writes, uniq)} writes with repeats, and only the most recent value bound to each key counts as correct. The delta rule recovers it with error <strong>${owDelta.toFixed(4)}</strong> — exactly zero, at any number of overwrites, as long as the distinct keys stay within capacity. The add-only rule is wrong by ${owAdd.toFixed(3)}, meaning the accumulated debris is ${owAdd >= 1 ? "larger than" : "comparable to"} the answer itself. The previous lab showed the wall both rules hit; this shows the thing only one of them can do, and the paper ran these as two separate experiments for exactly that reason. Both labs use β = ${fmt(beta, 2)} from the slider above.`;

    // --- the normalisation, on the real model
    const big = (mech) => {
      const s = forward(tokens, { mixer: stateMixer(mech) }).trace[0].heads[0].snapshots[Math.min(step, T) - 1];
      return Math.max(...s.flatMap((r) => Array.from(r, Math.abs)));
    };
    const bDelta = big({ write: "delta", beta });
    const bAdd = big({ write: "add" });
    const bRaw = big({ write: "delta", beta, sumNorm: false });
    liveRead.update({
      delta: fmt(bDelta, 2),
      add: fmt(bAdd, 2),
      raw: bRaw > 1e6 ? bRaw.toExponential(1) : fmt(bRaw, 2),
    });
    liveNote.className = "note " + (bRaw > 1e6 ? "warn" : "");
    liveNote.innerHTML = `The delta step is a gradient step, and its effective learning rate is the write strength times the squared length of the feature vector. It is stable only while that stays below two. On this model's real keys the squared length averages 33.9, so an unnormalised write overshoots its target by more than thirty times and alternates in sign: after ${Math.min(step, T)} token${step > 1 ? "s" : ""} the largest number in the state reaches <strong>${bRaw > 1e6 ? bRaw.toExponential(1) : fmt(bRaw, 2)}</strong>, against ${fmt(bDelta, 2)} with the paper's normalisation and ${fmt(bAdd, 2)} for the add rule. The fix is to divide each feature vector by the sum of its own components before anything touches it — and the paper derives that divisor rather than picking it: the sum is exactly the condition that the amount a write removes equals the amount it puts back. Its own words on running without it are that the models diverged. There is a second division this card does <em>not</em> do: the running denominator concept 9 carried is dropped here, because the paper measures it as a net negative for this rule and calls removing it crucial.`;
  }

  return { update: render, unmount: () => {} };
}
