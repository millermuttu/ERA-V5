// Concept 18 — YaRN.
// Built from docs/research/yarn.md. The research had to read three renderings of this paper because
// they are not the same paper: v1 (31 Aug 2023, the version this card is dated to) has no appendices,
// no ablation and no passkey experiment; v2 deleted v1's only explanation of the temperature; and the
// ablation that finally compares the method's two halves arrived in v3, in February 2026. So every
// quoted number on this card names its version. Four things that changed the build: YaRN contributes
// no interpolation of its own — Definition 3 says it is NTK-by-parts plus a temperature; the ramp is
// a per-dimension switch between plain interpolation and identity, which the seam could not express
// until `stretch` learned to be a function of the pair; the temperature is free only because it folds
// into the rotation's modulus, which is this timeline's one payoff for encoding position as a
// rotation; and the thing called a temperature measurably *sharpens* attention, which the card puts
// on screen rather than paraphrasing.
import { el, slider, choice, toggle } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout, barList } from "../views/bars.js";
import { fmt } from "../model/ops.js";
import { forward, DH } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { rope } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const B = 10000;
const PAIRS = DH / 2;
const WIDE = 128; // a realistic head, for the partition the app's four pairs cannot show

/** v2 eq. 13: the wavelength of pair i — how many tokens a full turn takes. */
const wavelength = (i, dims) => 2 * Math.PI * Math.pow(B, (2 * i) / dims);

/** v2 eq. 17: rotations completed inside the original context. The whole criterion is this ratio. */
const rotations = (i, dims, L) => L / wavelength(i, dims);

/** v2 eq. 18: 0 below α (interpolate fully), 1 above β (leave alone), linear crossfade between. */
const ramp = (r, a, b) => (r < a ? 0 : r > b ? 1 : (r - a) / (b - a));

/** v2 eq. 20: h(θ)/θ — a per-pair multiplier that is 1/s at γ=0 and 1 at γ=1. */
const byPartsFactor = (g, s) => (1 - g) / s + g;

/** Definition 3's other half, v2 eq. 22: √(1/t) = 0.1·ln(s) + 1, applied as the rotation's modulus. */
const tempFactor = (s) => 0.1 * Math.log(s) + 1;

const entropy = (row) => {
  let h = 0;
  for (const p of row) if (p > 0) h -= p * Math.log(p);
  return h;
};

export function yarnCard(root, m) {
  let L = 4096; // Llama 2's context, so the live split matches the one the note quotes
  let s = 16;
  let alpha = 1;
  let beta = 32;
  let pair = 0;
  let squeeze = false;
  let ablate = "both";

  root.appendChild(
    prose({
      problem:
        "Both earlier answers share a blind spot, and this paper names it in one word: blind. Interpolation divides every dimension by the same factor. The base change grades the factor by the dimension's index. Neither one asks the question that decides the case — does this dimension's wavelength actually fit inside the context the model was trained on? A dimension that turns a full circle every six tokens completed hundreds of turns during training: it has seen every angle there is, it carries relative offset, and squeezing it only makes neighbouring tokens harder to tell apart. A dimension whose wavelength is longer than the whole training context never completed a single turn: it carries absolute position, and it is the one that will be asked for angles that do not exist in the training data.",
      mechanism:
        "Compute the number of rotations each dimension completes inside the original context, and switch on it. Above a threshold, leave the dimension completely alone. Below a lower one, interpolate it exactly as the earlier method did. Between the two, cross-fade. That is the whole interpolation half, and it introduces nothing new — it is a per-dimension choice between the previous concept but one and doing nothing. The second half is a constant: multiply the queries and keys by a number slightly larger than one before the dot product. Because position here is a rotation, that constant can be folded into the rotation table itself, which was precomputed anyway — so a change to the attention softmax arrives without touching the attention code and at no cost per token.",
    })
  );

  root.appendChild(
    el("div", { class: "formula", text: "r(d) = L/λ_d      γ = clamp((r−α)/(β−α))      h(θ) = (1−γ)·θ/s + γ·θ      √(1/t) = 0.1·ln(s) + 1" })
  );

  root.appendChild(
    el("div", {
      class: "gate",
      html:
        "<strong>Which paper?</strong> This card is dated to <strong>v1</strong>, 31 August 2023 — the version with no appendices, no ablation, no passkey table and no evidence anywhere in it for its own abstract. v2 (Nov 2023) added most of the experiments and <em>deleted</em> v1's explanation of the temperature. The ablation that compares the method's two halves is <strong>v3, February 2026</strong>. Every quoted number below names its version; everything unquoted is this app's own arithmetic.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root);

  // ---------------------------------------------------- 1. the partition
  const lSlider = slider({
    label: "the context it was trained on",
    min: 4,
    max: 8192,
    step: 4,
    value: L,
    format: (v) => `${v} tokens`,
    oninput: (v) => ((L = v), render()),
  });
  const sSlider = slider({
    label: "extend by",
    min: 1,
    max: 32,
    step: 1,
    value: s,
    format: (v) => `${v}×`,
    oninput: (v) => ((s = v), render()),
  });
  const aSlider = slider({
    label: "α",
    min: 0.25,
    max: 8,
    step: 0.25,
    value: alpha,
    format: (v) => `${v}`,
    oninput: (v) => ((alpha = v), render()),
  });
  const bSlider = slider({
    label: "β",
    min: 2,
    max: 128,
    step: 1,
    value: beta,
    format: (v) => `${v}`,
    oninput: (v) => ((beta = v), render()),
  });
  const partBars = barList({
    rows: Array.from({ length: PAIRS }, (_, i) => ({ key: `p${i}`, label: `pair ${i}` })),
  });
  const partRead = readout([
    { key: "untouched", label: "left completely alone" },
    { key: "rampd", label: "in the crossfade" },
    { key: "full", label: "interpolated as before" },
    { key: "wide", label: "the same split on a 128-wide head" },
  ]);
  const partNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "which dimensions have a problem at all" }),
      el("div", { class: "ctrls" }, [lSlider.node, sSlider.node, aSlider.node, bSlider.node]),
      partBars.node,
      partRead.node,
      partNote,
    ])
  );

  // ------------------------------------------- 2. what the criterion means
  const pairPick = choice({
    label: "pair",
    options: Array.from({ length: PAIRS }, (_, i) => ({ value: String(i), label: `pair ${i}` })),
    value: String(pair),
    onchange: (v) => ((pair = Number(v)), render()),
  });
  const squeezeToggle = toggle({
    label: "apply the squeeze",
    value: false,
    onchange: (v) => ((squeeze = v), render()),
  });
  const angleCurve = curveView({
    xLabel: "position in the trained context",
    yLabel: "angle, wrapped to one turn",
    ariaLabel: "the rotation angle of one pair against position, wrapped at a full turn",
  });
  const angleRead = readout([
    { key: "lam", label: "this pair's wavelength" },
    { key: "r", label: "turns completed inside the context" },
    { key: "carries", label: "what it can encode" },
    { key: "gamma", label: "γ — how much of it survives" },
    { key: "shown", label: "positions drawn" },
  ]);
  const angleNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "unique angles, or the same angles over and over" }),
      el("div", { class: "ctrls" }, [pairPick, squeezeToggle]),
      angleCurve.node,
      angleRead.node,
      angleNote,
    ])
  );

  // ---------------------------------------- 3. three schemes on one axis
  const schemeCurve = curveView({
    xLabel: "dimension pair, fastest on the left",
    yLabel: "what its turn rate is multiplied by",
    ariaLabel: "the per-pair multiplier of all three schemes on one axis",
  });
  const schemeRead = readout([
    { key: "pi", label: "interpolation, every pair" },
    { key: "ntk", label: "the base change, fastest → slowest" },
    { key: "yarn", label: "this method, fastest → slowest" },
    { key: "bprime", label: "the base change's b′ at this width" },
  ]);
  const schemeNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the three answers, drawn on one axis" }),
      schemeCurve.node,
      schemeRead.node,
      schemeNote,
    ])
  );

  // --------------------------------------------- 4. the temperature
  const tempRead = readout([
    { key: "factor", label: "√(1/t) — the number q and k are multiplied by" },
    { key: "logit", label: "what that does to the logit" },
    { key: "norm", label: "the rotation's length, which RoPE kept at 1" },
    { key: "ent", label: "entropy of the last query's row, before → after" },
  ]);
  // The row of bars is one per token, so it is built on first render and rebuilt only when the
  // sentence changes length.
  let tempBars = null;
  let tempBarsLen = 0;
  const tempBarsHost = el("div", {});
  const tempNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "a temperature that folds into the rotation" }),
      tempRead.node,
      tempBarsHost,
      tempNote,
    ])
  );

  // ------------------------------------------------- 5. the ablation
  const ablatePick = choice({
    label: "run with",
    options: [
      { value: "off", label: "neither — plain rotation" },
      { value: "interp", label: "the interpolation only" },
      { value: "temp", label: "the temperature only" },
      { value: "both", label: "both — this is the method" },
    ],
    value: ablate,
    onchange: (v) => ((ablate = v), render()),
  });
  const ablateRead = readout([
    { key: "moved", label: "how far the attention moved from plain rotation" },
    { key: "ent", label: "entropy of the last query's row" },
    { key: "quotedRaw", label: "quoted: perplexity at 16×, no fine-tuning" },
    { key: "quotedFt", label: "quoted: the same, fine-tuned 400 steps" },
  ]);
  const ablateNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what each half does on its own" }),
      el("div", { class: "ctrls" }, [ablatePick]),
      ablateRead.node,
      ablateNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "It asks the only question that distinguishes the cases — does this dimension's wavelength fit inside the trained context — and treats the two answers differently, where both earlier methods applied one rule to the whole ladder",
        "The dimensions that already turn many times inside the context are left exactly as trained, so nothing is done to the resolution that separates adjacent tokens",
        "The dimensions whose wavelength exceeds the context are interpolated outright, so nothing is pushed past an angle the model never saw — the previous concept's out-of-bound problem is driven to zero by construction",
        "The temperature costs nothing to run: because position is a rotation, the constant folds into the precomputed rotation table and the attention code never changes",
        "It reaches its result with about 0.1% of a pre-training corpus — 400 steps, roughly 1.7 billion tokens, and 128 to 384 A100-hours against Code Llama's 6,400 (v3 Table 4)",
      ],
      givesUp: [
        "The two thresholds that decide how a third of a real model's dimensions are treated are unswept constants — α = 1 and β = 32, with “tuned on a case-by-case basis” as the whole methodology and no argument anywhere for 32 rather than 8 or 128",
        "The temperature has no derivation. v1 gave an entropy argument that contradicts itself inside one sentence, v2 deleted it and left a curve fit, and the data that fit was made from is printed in no version — only figures",
        "It is called a temperature and it does the opposite of what that word implies: the recommended factor is greater than one, so the logits are multiplied up and the attention gets sharper, not flatter",
        "The halves are not cleanly additive. Without fine-tuning the interpolation alone collapses and the temperature carries the result; with fine-tuning the interpolation alone is better at the original context and the temperature is a small tax (v3 Table 5)",
        "Graceful degradation past the extended window is not its property — the fixed-scale models blow past 10 in perplexity beyond their window exactly as interpolation does. That property belongs to dynamic scaling, which is orthogonal and applies to every method here",
        "Every headline number is a perplexity, and the paper's own appendix suspects perplexity of not measuring the thing it is being used to claim",
      ],
      chooseWhen:
        "The default for extending a rotary model that you can afford to fine-tune briefly — it is what shipped, and the drop on the original context is a point or two rather than the collapse the base change produced. If you cannot fine-tune at all, the temperature half is what is carrying the result and you should keep it; if you can, and you only care about the original context, the interpolation half alone measures slightly better.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "It sorts the parts of the mechanism into ones that have a problem and ones that do not, instead of treating all of them the same",
        "The parts that tell neighbouring words apart are left exactly as they were, and the parts that were never fully used take the whole squeeze",
        "The second half of the method is a single multiplication that rides along inside a table the model already builds, so it costs nothing to run",
        "The whole thing was reached with a few hundred training steps on a corner of a training budget, rather than a fresh run",
      ],
      cons: [
        "Two numbers decide how most of the mechanism is treated, and nobody has published what happens if you pick different ones",
        "The second half is named after an effect that is the reverse of what it measurably does, and the reasoning behind it was withdrawn from the paper rather than repaired",
        "Which half is doing the work flips depending on whether you can afford further training, and the comparison that shows this appeared two and a half years after the method",
        "Everything is measured by one score that the authors themselves suspect does not capture whether the model can actually use the extra length",
      ],
      verdict:
        "The end of the extension argument, and it wins it by changing the question. Not how hard to squeeze, but which parts have a squeezing problem at all — a dimension that has already been round the circle hundreds of times inside the training text needs nothing done to it, and one that never got round once needs all of it. Bolted to that is a free constant that the field calls a temperature, that sharpens rather than softens, and that only works at all because someone decided two years earlier to encode position as a rotation. What it leaves behind is the honest doubt in its own appendix: every number here measures how surprised the model is by text, and being unsurprised at 128,000 tokens is not the same as being able to use them.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    lSlider.set(L);
    sSlider.set(s);
    aSlider.set(alpha);
    bSlider.set(beta);

    const hi = Math.max(alpha + 0.25, beta);
    const gammaOf = (i, dims = DH, ctx = L) => ramp(rotations(i, dims, ctx), alpha, hi);
    const byParts = (i) => byPartsFactor(gammaOf(i), s);
    const tf = tempFactor(s);

    const schemes = {
      off: rope({ dims: DH }),
      interp: rope({ dims: DH, stretch: byParts }),
      temp: rope({ dims: DH, modulus: tf }),
      both: rope({ dims: DH, stretch: byParts, modulus: tf }),
    };
    const run = (r) => forward(tokens, { mixer: softmaxMixer({ rotate: (v, p) => r.rotate(v, p) }) });
    const runs = Object.fromEntries(Object.entries(schemes).map(([k, r]) => [k, run(r)]));
    const rowOf = (res) => res.trace[0].heads[0].weights[T - 1];
    const maxDiff = (a, b) => {
      let d = 0;
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) d = Math.max(d, Math.abs(a[i][j] - b[i][j]));
      return d;
    };
    const weightsOf = (res) => res.trace[0].heads[0].weights;

    const active = runs[ablate];
    const h = active.trace[0].heads[0];
    flow.update({
      tokens,
      head: { ...h, emb: active.trace[0].input },
      weights: h.weights,
      out: h.out,
      top: active.top,
      query: T - 1,
      opts: {
        rotate: {
          angle: (pos) => pos * schemes[ablate].applied(0),
          label: ablate === "temp" || ablate === "both" ? "turned by position, and lengthened" : "turned by position",
        },
      },
    });
    flowNote.innerHTML = `Still concept 12's picture, and still only two knobs on it. What this method changes is that the turn rate is now a <em>different multiplier for each pair</em> rather than one number for all of them, and that the rotation is allowed a length other than 1. Nothing in the attention code below the dial has moved — which is the paper's own claim for the second half: it <em>“can effectively alter the attention mechanism without modifying its code.”</em>`;

    // --- 1. the partition
    let counts = { untouched: 0, rampd: 0, full: 0 };
    const rows = {};
    for (let i = 0; i < PAIRS; i++) {
      const r = rotations(i, DH, L);
      const g = gammaOf(i);
      const band = g === 1 ? "untouched" : g === 0 ? "full" : "rampd";
      counts[band]++;
      rows[`p${i}`] = {
        value: g,
        of: 1,
        text: `λ ${fmt(wavelength(i, DH), 1)} · r ${r >= 0.01 ? fmt(r, 2) : r.toExponential(1)} · γ ${fmt(g, 3)}`,
        tone: g === 1 ? "" : g === 0 ? "alt" : "",
      };
    }
    partBars.update(rows);
    const wide = { untouched: 0, rampd: 0, full: 0 };
    for (let i = 0; i < WIDE / 2; i++) {
      const g = gammaOf(i, WIDE, L);
      wide[g === 1 ? "untouched" : g === 0 ? "full" : "rampd"]++;
    }
    partRead.update({
      untouched: `${counts.untouched} of ${PAIRS}`,
      rampd: `${counts.rampd} of ${PAIRS}`,
      full: `${counts.full} of ${PAIRS}`,
      wide: `${wide.untouched} / ${wide.rampd} / ${wide.full}`,
    });
    const degenerate = counts.untouched === 0;
    partNote.className = "note " + (degenerate ? "warn" : "good");
    partNote.innerHTML = `Each bar is γ, the fraction of a pair's turn rate that survives: a full bar is a dimension left exactly as trained, an empty ochre bar is one squeezed by the full ${fmt(
      s,
      0
    )}×, and anything between is the crossfade. The rule behind it is one ratio — how many complete turns the pair makes inside the trained context — and the two thresholds the paper recommends, <em>“good values for α and β are α = 1 and β = 32”</em>, are the sliders above. ${
      degenerate
        ? `<strong>At a ${L}-token context nothing is left alone</strong>, so with only ${PAIRS} pairs this method has degenerated into plain interpolation. That is honest rather than broken: no pair here completes ${fmt(
            hi,
            0
          )} turns in that little text. Push the context up and watch the top bar fill.`
        : `At ${L} tokens, ${counts.untouched} of ${PAIRS} pairs are left completely alone and ${counts.full} take the full squeeze.`
    } The app has four pairs, which is too few to see the shape, so the last readout runs the same arithmetic on a 128-wide head: <strong>${
      wide.untouched
    } untouched, ${wide.rampd} in the crossfade, ${
      wide.full
    } fully interpolated</strong>. At the paper's own settings and a 4,096-token context that is 21 / 25 / 18 — a third left alone, a third squeezed, and a crossfade bigger than either. The paper prints that split nowhere, and it is entirely determined by two numbers it never sweeps.`;

    // --- 2. what the criterion means
    const th = 1 / Math.pow(B, (2 * pair) / DH);
    const rate = squeeze ? th * byPartsFactor(gammaOf(pair), s) : th;
    // Draw at most ~40 turns: past that the teeth are narrower than the sampling and the picture
    // becomes an aliasing artefact rather than the pair's actual behaviour.
    const lamHere = wavelength(pair, DH);
    const span = Math.min(L, Math.max(64, 40 * lamHere));
    const step = span / 600;
    const pts = [];
    for (let p = 0; p <= span; p += step) {
      const a = (p * rate) % (2 * Math.PI);
      // A break at each wrap, so the sawtooth reads as teeth rather than one zig-zag line.
      if (pts.length && a < pts[pts.length - 1][1]) pts.push([p, undefined]);
      pts.push([p, a]);
    }
    const lam = lamHere;
    const r = rotations(pair, DH, L);
    const g = gammaOf(pair);
    angleCurve.update({
      points: pts,
      xRange: [0, span],
      yRange: [0, 2 * Math.PI],
      band: lam < span ? [0, lam] : null,
      mark: lam < span ? lam : null,
      markLabel: "one full turn",
      deadLabel: "",
    });
    angleRead.update({
      lam: `${fmt(lam, 1)} tokens`,
      r: r >= 0.01 ? fmt(r, 2) : r.toExponential(1),
      carries: r < 1 ? "absolute position" : "relative offset",
      gamma: fmt(g, 3),
      shown: span < L ? `first ${fmt(span, 0)} of ${L}` : `all ${L}`,
    });
    angleNote.className = "note";
    angleNote.innerHTML =
      r < 1
        ? `This pair takes ${fmt(
            lam,
            1
          )} tokens to complete one turn, and the context is only ${L}, so it never gets there: the line rises once and stops. Every position in the whole context has its own angle, never repeated — <em>“having all unique position pairs implies that the absolute positional information remains intact.”</em> This is the kind of dimension that must be interpolated, because asking it for a position ${fmt(
            s,
            0
          )}× further out means asking it for angles it has never once produced. γ is ${fmt(
            g,
            3
          )}: it takes the full squeeze.${
            squeeze ? ` With the squeeze on you can see the line rise more gently — that is the whole effect on this pair, and it costs nothing.` : ""
          }`
        : `This pair turns a full circle every ${fmt(
            lam,
            1
          )} tokens and gets round ${fmt(
            r,
            1
          )} times inside the context, so the same angles come back again and again — <em>“when the wavelength is short, only relative positional information is accessible to the network.”</em> Nothing here is unexplored, so nothing needs interpolating; γ is ${fmt(
            g,
            3
          )}. ${
            squeeze
              ? g < 0.999
                ? `<span class="warn">With the squeeze on, the teeth are wider</span> — fewer, slower turns across the same text, which is exactly the damage the paper describes: <em>“the model being confused on the positional order of close-by tokens.”</em> Turn β down until γ reaches 1 and the teeth snap back.`
                : `With the squeeze on, nothing changes — γ is 1, so this pair is passed through untouched. That is the point of the method.`
              : `Switch the squeeze on and compare: under plain interpolation these teeth would get ${fmt(
                  s,
                  0
                )}× wider, which is the resolution this pair exists to provide.`
          }`;

    // --- 3. three schemes on one axis
    const nPairs = WIDE / 2;
    const bPrime = B * Math.pow(s, WIDE / (WIDE - 2));
    const piCurve = [];
    const yarnCurve = [];
    const ntkDots = [];
    for (let i = 0; i < nPairs; i++) {
      piCurve.push([i, 1 / s]);
      yarnCurve.push([i, byPartsFactor(gammaOf(i, WIDE, L), s)]);
      ntkDots.push([i, Math.pow(s, (-2 * i) / (WIDE - 2))]);
    }
    schemeCurve.update({
      points: yarnCurve,
      reference: piCurve,
      xRange: [0, nPairs - 1],
      yRange: [0, 1.18],
    });
    schemeCurve.setDots(
      ntkDots.filter((_, i) => i % 2 === 0).map(([x, y]) => [x, y, "ref"]),
      [0, nPairs - 1],
      [0, 1.18]
    );
    const appYarn = Array.from({ length: PAIRS }, (_, i) => fmt(byPartsFactor(gammaOf(i), s), 3)).join(" ");
    const appNtk = Array.from({ length: PAIRS }, (_, i) => fmt(Math.pow(s, (-2 * i) / (DH - 2)), 3)).join(" ");
    schemeRead.update({
      pi: fmt(1 / s, 4),
      ntk: appNtk,
      yarn: appYarn,
      bprime: bPrime.toExponential(2),
    });
    schemeNote.className = "note";
    schemeNote.innerHTML = `Three answers to the same question, on a 128-wide head so the shape is visible. The dashed line is interpolation: <strong>flat at ${fmt(
      1 / s,
      4
    )}</strong>, every dimension treated identically, blind. The grey dots are the base change: a smooth slide from 1 down to ${fmt(
      1 / s,
      4
    )}, graded — but graded by the dimension's <em>index</em>, which is still blind to whether that dimension had a problem. The solid line is this method: exactly 1 on the left where dimensions turn many times inside the context, exactly ${fmt(
      1 / s,
      4
    )} on the right where they never complete a turn, and a straight crossfade between. Only the third curve is a function of the wavelength relative to the context, and that is the paper's own distinction between <em>blind</em> and <em>targeted</em> methods. One caution about the app's own numbers: at ${DH} dimensions per head the base change's exponent is 1.3333 rather than a real head's 1.0159, so this app's four-pair version of the middle curve is a third steeper than the real thing.`;

    // --- 4. the temperature
    const plainRow = rowOf(runs.off);
    const tempRow = rowOf(runs.temp);
    const hBefore = entropy(plainRow);
    const hAfter = entropy(tempRow);
    const q = runs.off.trace[0].heads[0].Q[T - 1];
    const rotated = schemes.temp.rotate(q, T - 1);
    const norm = Math.sqrt(rotated.reduce((a, x) => a + x * x, 0)) / Math.sqrt(q.reduce((a, x) => a + x * x, 0));
    tempRead.update({
      factor: fmt(tf, 4),
      logit: `×${fmt(tf * tf, 4)}`,
      norm: fmt(norm, 4),
      ent: `${fmt(hBefore, 4)} → ${fmt(hAfter, 4)}`,
    });
    if (tempBarsLen !== T) {
      tempBars = barList({ rows: tokens.map((t, i) => ({ key: `k${i}`, label: `${i}  ${t.word}` })) });
      tempBarsLen = T;
      tempBarsHost.replaceChildren(tempBars.node);
    } else {
      tempBars.node.querySelectorAll(".bar-name").forEach((n, i) => (n.textContent = `${i}  ${tokens[i].word}`));
    }
    const barVals = {};
    for (let i = 0; i < T; i++) {
      barVals[`k${i}`] = {
        value: tempRow[i],
        of: Math.max(...tempRow),
        text: `${fmt(tempRow[i], 3)} ← ${fmt(plainRow[i], 3)}`,
        tone: tempRow[i] > plainRow[i] ? "" : "alt",
      };
    }
    tempBars.update(barVals);
    tempNote.className = "note warn";
    tempNote.innerHTML = `RoPE multiplies q and k by numbers of length exactly 1, which is why concept 12 could report that the rotation preserves the vector's length. Give those numbers length <strong>${fmt(
      tf,
      4
    )}</strong> instead — the readout above is that length, measured on the model's own query — and the dot product comes out multiplied by ${fmt(
      tf * tf,
      4
    )}. That is a softmax temperature, obtained by editing a table that was precomputed anyway, with no change to the attention code and no cost per token. <strong>It is available here and nowhere else on this timeline</strong>: an additive position encoding cannot be folded into a scale, and adding a constant to logits does nothing to a softmax at all. Now read the direction. The bars are the last query's attention with the constant applied, each showing what it was before; entropy goes <strong>${fmt(
      hBefore,
      4
    )} → ${fmt(
      hAfter,
      4
    )}</strong> nats, against ${fmt(
      Math.log(T),
      4
    )} for a flat row over ${T} tokens. The recommended factor is greater than one for every extension, so it always <em>sharpens</em>. The paper calls it a temperature and v1 motivated it as <em>“increase the ‘temperature’ of the attention logits”</em> — v1's own explanation contradicts itself inside a single sentence, and v2 deleted the explanation and kept the constant. Note also that v1 wrote the fit as √t = 0.1 ln(s) + 1 and v2 as √(1/t) = 0.1 ln(s) + 1: a reciprocal symbol flip that changed no implementation, because both name the same multiplier.`;

    // --- 5. the ablation
    const moved = maxDiff(weightsOf(runs.off), weightsOf(active));
    const quoted = {
      off: { raw: "4.05 · unextended", ft: "—" },
      interp: { raw: "6.04 → >10¹", ft: "4.14 / 3.62" },
      temp: { raw: "not measured", ft: "not measured" },
      both: { raw: "4.61 → 3.45", ft: "4.19 / 3.30" },
    }[ablate];
    ablateRead.update({
      moved: ablate === "off" ? "—" : fmt(moved, 3),
      ent: fmt(entropy(rowOf(active)), 4),
      quotedRaw: quoted.raw,
      quotedFt: quoted.ft,
    });
    ablateNote.className = "note";
    ablateNote.innerHTML = `The two live numbers are this app's, on your sentence; the two quoted ones are from <strong>Table 5 of v3</strong>, LLaMA 7B extended 16×, and no version before February 2026 contains them. Read them together and the usual telling falls apart. <strong>Without fine-tuning, the interpolation half alone collapses</strong> — 6.04 at the original length and past 10 further out — while the same interpolation plus the constant holds at 4.61 and improves to 3.45. In that regime the temperature is not a garnish; it is carrying the entire result, and the interpolation half is the second-worst method in the table. <strong>After 400 steps of fine-tuning it reverses</strong>: the interpolation alone is better at the original context (4.14 against 4.19) and the full method wins at long range (3.30 against 3.62). So the honest decomposition is that the interpolation is what survives training and the temperature is what makes the method work without it — at a small cost near the original context. The paper never comments on either comparison, and its own conclusion — <em>“with no downsides”</em> — is contradicted by cells in three of its tables. The temperature-only column is empty because nobody has ever measured it.`;
  }

  return { update: render, unmount: () => {} };
}
