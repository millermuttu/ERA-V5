// Concept 17 — NTK-aware scaled RoPE.
// Built from docs/research/ntk-aware.md. Four things the research settled, and each of them shapes
// a panel. The seam needs nothing new: PI is `rope({stretch})` and this is `rope({base})`, and both
// knobs have been sitting in position.js since concept 12. The exponent is not derived from neural
// tangent kernel theory despite the name — YaRN Appendix A.1 derives it from a wavelength boundary
// condition on the *slowest* pair, and the fastest pair is left alone for free by the shape of the
// ladder. The famous defect ("some dimensions go out-of-bound") is asserted in YaRN with no count,
// no plot and no threshold, so the counts on this card are this app's arithmetic from the
// definitions. And the primary source is a Reddit post whose entire empirical content is three
// images: every perplexity claim attached to this method traces to pictures, on a model this app
// cannot run. So the card measures geometry, which it can settle exactly, and attributes quality,
// which it cannot touch at all.
import { el, slider, choice } from "../lib/dom.js";
import { curveView } from "../views/curve.js";
import { readout, barList } from "../views/bars.js";
import { fmt } from "../model/ops.js";
import { forward, DH } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { rope } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

const B = 10000;
const EXP = DH / (DH - 2); // 8/6 at this app's head width; 128/126 on a real head

/** YaRN Definition 1, eq. 16: b' = b · s^(|D|/(|D|−2)). One constant, one pow. */
const newBase = (s, dims = DH) => B * Math.pow(s, dims / (dims - 2));

/** θ_i = base^(−2i/|D|), the same ladder position.js builds, at an arbitrary width. */
const freqsAt = (base, dims) => Array.from({ length: dims / 2 }, (_, i) => 1 / Math.pow(base, (2 * i) / dims));

/** Eq. 23 read as a ratio: pair i is compressed by s^(2i/(|D|−2)) — 1 at the fast end, s at the slow. */
const effScale = (s, i, dims) => Math.pow(s, (2 * i) / (dims - 2));

/** What is left over: the angle at the extended end divided by the largest angle seen in training. */
const overshoot = (s, i, dims) => s / effScale(s, i, dims);

const PAIRS = DH / 2;
const SAMPLE_128 = [0, 8, 16, 32, 41, 48, 56, 63];

export function ntkAwareCard(root, m) {
  let s = 4;
  let width = "8";
  let alpha = 4;

  root.appendChild(
    prose({
      problem:
        "The previous concept works and its proof is real, but it charges one price to every frequency. Divide every position by four and the pair that turns a full circle every six tokens — the one telling adjacent words apart — loses three quarters of its resolution, while the pair with a wavelength of six thousand, which never came close to completing a turn during training, loses the same three quarters it could easily have afforded. That card ended on exactly this question: why is the slowest pair paying the same tax as the fastest? Two days after the interpolation paper appeared, somebody on a forum answered it.",
      mechanism:
        "Leave the positions alone and change how fast each pair turns. Every rotation rate comes from one constant, the base, raised to a power that depends on the pair's index — so raising the base slows the whole ladder down, and it slows the slow pairs far more than the fast ones. Pick the new base so the slowest pair ends up exactly where interpolation would have put it, and the fastest is left untouched automatically, because any base raised to the power zero is one. Everything between the two ends is a geometric blend. The whole change is one number inside one pow, and no position is rescaled, no dimension special-cased, no term added to the score.",
    })
  );

  root.appendChild(el("div", { class: "formula", text: "b′ = b · s^(|D|/(|D|−2))          θ′_i = b′^(−2i/|D|)          s_i = θ_i/θ′_i = s^(2i/(|D|−2))" }));

  const { flow, note: flowNote } = flowPanel(root);

  // ---------------------------------------------- 1. one dial, two knobs
  const sSlider = slider({
    label: "extend the context by",
    min: 1,
    max: 32,
    step: 1,
    value: s,
    format: (v) => `${v}×`,
    oninput: (v) => ((s = v), render()),
  });
  const ladderBars = barList({
    rows: Array.from({ length: PAIRS }, (_, i) => [
      { key: `pi${i}`, label: `pair ${i} · interpolation`, alt: true },
      { key: `ntk${i}`, label: `pair ${i} · base change` },
    ]).flat(),
  });
  const ladderRead = readout([
    { key: "base", label: "the new base b′" },
    { key: "exp", label: "the exponent |D|/(|D|−2)" },
    { key: "ntkscales", label: "what each pair is actually compressed by" },
    { key: "piscales", label: "what interpolation compresses them by" },
  ]);
  const ladderNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "one dial, two knobs — how far each pair moves" }),
      el("div", { class: "ctrls" }, [sSlider.node]),
      ladderBars.node,
      ladderRead.node,
      ladderNote,
    ])
  );

  // ------------------------------------------- 2. the neighbour test
  const neighbourRead = readout([
    { key: "fastPi", label: "angle between neighbours, fastest pair — interpolation" },
    { key: "fastNtk", label: "angle between neighbours, fastest pair — base change" },
    { key: "movedPi", label: "how far the attention moved — interpolation" },
    { key: "movedNtk", label: "how far the attention moved — base change" },
  ]);
  const neighbourNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the neighbour test — why this one needs no fine-tuning" }),
      neighbourRead.node,
      neighbourNote,
    ])
  );

  // ------------------------------------- 3. which dimensions go out of bounds
  const widthPick = choice({
    label: "head width",
    options: [
      { value: "8", label: "8 — this app" },
      { value: "128", label: "128 — a real head" },
    ],
    value: width,
    onchange: (v) => ((width = v), render()),
  });
  const boundBars8 = barList({
    rows: Array.from({ length: PAIRS }, (_, i) => ({ key: `p${i}`, label: `pair ${i}` })),
  });
  const boundBars128 = barList({
    rows: SAMPLE_128.map((i) => ({ key: `p${i}`, label: `pair ${i}` })),
  });
  const boundRead = readout([
    { key: "over", label: "pairs ending past their trained angle" },
    { key: "never", label: "pairs that never complete a turn in training" },
    { key: "both", label: "both at once — the ones that matter" },
  ]);
  const boundNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "which dimensions end up outside what they were trained on" }),
      el("div", { class: "ctrls" }, [widthPick]),
      boundBars8.node,
      boundBars128.node,
      boundRead.node,
      boundNote,
    ])
  );

  // ----------------------------------------- 4. the dynamic variant
  const alphaSlider = slider({
    label: "requested α",
    min: 2,
    max: 16,
    step: 1,
    value: alpha,
    format: (v) => `${v}×`,
    oninput: (v) => ((alpha = v), render()),
  });
  const dynCurve = curveView({
    xLabel: "length of the sequence being processed",
    yLabel: "effective base",
    ariaLabel: "the base recomputed per forward pass, shipped ramp against the paper's",
  });
  const dynRead = readout([
    { key: "asked", label: "α you asked for" },
    { key: "got", label: "α the shipped ramp gives at that length" },
    { key: "short", label: "attention change on your sentence, dynamic" },
    { key: "shortStatic", label: "attention change, static at this scale" },
  ]);
  const dynNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "recompute the scale from the length you actually have" }),
      el("div", { class: "ctrls" }, [alphaSlider.node]),
      dynCurve.node,
      dynRead.node,
      dynNote,
    ])
  );

  // ------------------------------------------------- 5. provenance
  const claim = (text, tone = "") => el("div", { class: "gate" + (tone ? " " + tone : ""), html: text });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what was actually checked, and by whom" }),
      claim(
        "<strong>The base formula.</strong> Claimed by bloc97 in the post, with no derivation — <em>“I might give a more detailed explanation on how I derived the formula used to calculate the base if enough people are interested.”</em> The only published account is YaRN Appendix A.1, three sentences long, by the same author two months later. For those two months the method was adopted into <code>transformers</code>, Code Llama and Qwen with no derivation in public at all. <strong>Reproducible arithmetic — every number in the panels above is this app recomputing it.</strong>"
      ),
      claim(
        "<strong>“Minimal perplexity degradation.”</strong> Claimed by bloc97. The evidence is three images: LLaMA 7B on 40 prompts from a subset of gov_report. No table, no perplexity value in the text, no baseline number, no variance, no seed, no second model. <strong>Attributed only — this app cannot compute perplexity and cannot inspect the images.</strong>"
      ),
      claim(
        "<strong>“No fine-tuning needed.”</strong> Claimed by bloc97, who adds: <em>“I did not test fine-tuning performance as I do not have the resources or the time to fine tune an LLM, I just derived this formula during lunch and experimented with it. However, I think that this method will do even better with fine tuning.”</em> That last guess turned out backwards. <strong>Attributed only.</strong>"
      ),
      claim(
        "<strong>“Fine-tuning with base scaling is worse than interpolation.”</strong> YaRN §3.1 and Table 1: about 0.25 perplexity worse at every length inside the extended window, and much better past it — 6.24 against 8.07 at 10,240. One model pair, one extension, ten documents, one run per arm, no variance, and the two arms were fine-tuned by different people. Self-reported by the same authors, in a paper proposing the replacement. <strong>Attributed only.</strong>",
        "is-blocked"
      ),
      claim(
        "<strong>“Some dimensions are slightly extrapolated to out-of-bound values.”</strong> YaRN §3.1, asserted with no count, no plot, no threshold and no per-dimension breakdown. <strong>This app measures exactly which ones</strong> — the panel above is that measurement, from the definitions rather than from the paper."
      ),
      el("p", {
        class: "note",
        html:
          "Two of five rows this app can verify itself; three it can only attribute. That ratio is the entry. It is also the only concept on this timeline whose primary source is a forum post — which is why the badge above says unverified and why the date needed archive.org to pin down: the post is <strong>29 June 2023, 08:21 UTC</strong>, and the widely-cited 30 June is the <em>follow-up</em> post, by the other author, introducing the dynamic variant.",
      }),
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Adjacent tokens stay exactly as distinguishable as they were: the fastest pair's angle between neighbours is unchanged at 1.000 radian, where interpolation divides it by the full extension factor",
        "The trained model is left intact — one constant changes, no weight moves, no position is rescaled, and the shipped change is a single line",
        "It needs no fine-tuning to be usable, which is why it reached half the open-source ecosystem within three weeks of a forum post",
        "It is still exactly relative: the base change is a different rotation schedule, not a different kind of encoding, so shifting the whole sentence still leaves the attention matrix identical to floating-point noise",
        "In the dynamic form the model below its trained length is bit-for-bit the original, so short sequences pay nothing at all",
      ],
      givesUp: [
        "Every pair except the last ends at an angle larger than any angle it saw in training — at 128 dimensions that is 63 of 64 pairs, and the 22 that both overshoot and never completed a turn during training are the ones YaRN means by out-of-bound",
        "After fine-tuning it loses to plain interpolation at every length inside the extended window, by about 0.25 perplexity in the only published side-by-side — the author's own guess that fine-tuning would help it most was wrong",
        "The requested scale under-delivers: every pair but the last is compressed by less than the factor you asked for, so in practice s must be set higher than the extension you actually want",
        "The name promises a derivation it does not have. No kernel is computed anywhere; the exponent comes from a wavelength boundary condition, and the authors call the base change merely the simplest transformation with those endpoints",
        "Its entire empirical case is three images on 40 prompts of one model, and nothing outside the LLaMA family was ever measured",
      ],
      chooseWhen:
        "When a trained model already exists, already uses rotary positions, and needs a longer context tonight without a fine-tuning run — especially in the dynamic form, which costs nothing below the trained length. If you can afford to fine-tune, the measured result says use interpolation instead, or better, the method that repairs both.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "Words that sit next to each other stay just as easy to tell apart as before, which is the thing the previous method broke",
        "Nothing about the trained model changes — one setting is different, and it works immediately, with no further training",
        "Moving the whole passage further into a document still changes nothing, exactly as before",
        "In the version that adjusts itself as it reads, short passages are handled exactly as the original model would have handled them",
      ],
      cons: [
        "Almost every part of the mechanism ends up turned further than it was ever turned while the model was learning, which is a place nobody has checked",
        "If you can afford further training, the older and simpler method measures better inside the range you asked for",
        "Ask for four times the length and you get less than four times, so people quietly ask for more than they need",
        "The name suggests a piece of theory that was never actually used, and the evidence behind the original claim is three pictures of one model",
      ],
      verdict:
        "The first idea on this timeline to notice that a rotation's frequencies are not all alike — the fast ones carry word order and cannot be squeezed, the slow ones were never used up and can be squeezed freely — and to move all of them by one constant in exactly that graded way. It was written on a forum in an afternoon, adopted everywhere within weeks, and is wrong in a productive way: the grading is right, the particular curve is arbitrary, and almost every dimension ends up a little outside where it was trained. The next entry is those two defects being repaired one at a time.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    sSlider.set(s);
    alphaSlider.set(alpha);

    // The app has no trained length, so the sentence stands in for one: L is what the model has
    // "seen", N = s·L is where it is being asked to read. That substitution is the card's one
    // pretence and the note says so.
    const L = T;
    const bp = newBase(s);
    const plain = rope({ base: B, dims: DH });
    const pi = rope({ base: B, stretch: 1 / s, dims: DH });
    const ntk = rope({ base: bp, dims: DH });

    const matrix = (r, shift = 0) =>
      forward(tokens, { mixer: softmaxMixer({ rotate: (v, p) => r.rotate(v, p + shift) }) }).trace[0].heads[0].weights;
    const maxDiff = (a, b) => {
      let d = 0;
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) d = Math.max(d, Math.abs(a[i][j] - b[i][j]));
      return d;
    };
    const baseline = matrix(plain);
    const movedPi = maxDiff(baseline, matrix(pi));
    const movedNtk = maxDiff(baseline, matrix(ntk));

    const res = forward(tokens, { mixer: softmaxMixer({ rotate: (v, p) => ntk.rotate(v, p) }) });
    const h = res.trace[0].heads[0];
    flow.update({
      tokens,
      head: { ...h, emb: res.trace[0].input },
      weights: h.weights,
      out: h.out,
      top: res.top,
      query: T - 1,
      opts: {
        rotate: { angle: (pos) => pos * ntk.freqs[0], label: "Q and K turned — fastest pair unchanged" },
      },
    });
    flowNote.innerHTML = `The picture is concept 12's, unchanged, and that is the claim: nothing here is a new operation. The queries and keys are still turned before the dot product, the values are still left alone, and the only difference from plain rotation is the number the turn rates are built from — <strong>${fmt(
      bp,
      2
    )}</strong> instead of ${B.toLocaleString()}. At the extension you have chosen, the fastest pair's rate is identical to plain rotation, so on a short sentence like this one the fast structure of the picture is exactly what it was.`;

    // --- 1. the ladder
    const piLam = plain.freqs.map((f) => (2 * Math.PI) / (f / s));
    const ntkLam = ntk.freqs.map((f) => (2 * Math.PI) / f);
    const top = Math.log10(Math.max(...piLam, ...ntkLam));
    const bars = {};
    for (let i = 0; i < PAIRS; i++) {
      const sc = effScale(s, i, DH);
      bars[`pi${i}`] = { value: Math.log10(piLam[i]), of: top, text: `${fmt(piLam[i], 1)}  ×${fmt(s, 2)}` };
      bars[`ntk${i}`] = {
        value: Math.log10(ntkLam[i]),
        of: top,
        text: `${fmt(ntkLam[i], 1)}  ×${fmt(sc, 3)}`,
      };
    }
    ladderBars.update(bars);
    ladderRead.update({
      base: fmt(bp, 2),
      exp: fmt(EXP, 4),
      ntkscales: plain.freqs.map((_, i) => fmt(effScale(s, i, DH), 3)).join("  "),
      piscales: plain.freqs.map(() => fmt(s, 3)).join("  "),
    });
    ladderNote.className = "note " + (s > 1 ? "good" : "");
    ladderNote.innerHTML =
      s === 1
        ? `At 1× nothing is extended and the two schemes are the same as plain rotation. Move the dial and watch the top pair of bars: under interpolation every pair's wavelength is multiplied by the same factor, under the base change the first pair does not move at all.`
        : `Read the first and last pairs. <strong>Pair 0 does not move</strong> under the base change — its wavelength stays ${fmt(
            ntkLam[0],
            1
          )} where interpolation stretches it to ${fmt(
            piLam[0],
            1
          )} — and <strong>pair ${PAIRS - 1} lands exactly where interpolation put it</strong>, ${fmt(
            ntkLam[PAIRS - 1],
            1
          )} against ${fmt(
            piLam[PAIRS - 1],
            1
          )}. That is not a coincidence, it is the specification: YaRN's eq. 23 asks for one thing only — make the slowest pair's new wavelength exactly ${fmt(
            s,
            0
          )} times its old one — and the fast end comes out fixed for free, because any base to the power zero is one. Everything between is a geometric blend, so the compression row reads <em>${plain.freqs
            .map((_, i) => fmt(effScale(s, i, DH), 3))
            .join(", ")}</em> against interpolation's flat <em>${plain.freqs.map(() => fmt(s, 0)).join(", ")}</em>. One ramp against one flat row is the whole mechanism. Note the exponent as well: at ${DH} dimensions per head it is ${fmt(
            EXP,
            4
          )}, so b′ is ${fmt(
            Math.pow(s, EXP) / s,
            2
          )}× further than the naive b·s — on a real 128-wide head the same exponent is 1.0159 and that correction almost vanishes.`;

    // --- 2. the neighbour test
    const fastPlain = plain.freqs[0];
    const slowPlain = plain.freqs[PAIRS - 1];
    neighbourRead.update({
      fastPi: `${fmt(fastPlain / s, 4)} rad`,
      fastNtk: `${fmt(ntk.freqs[0], 4)} rad`,
      movedPi: fmt(movedPi, 3),
      movedNtk: fmt(movedNtk, 3),
    });
    neighbourNote.className = "note";
    neighbourNote.innerHTML = `Plain rotation turns the fastest pair by <strong>${fmt(
      fastPlain,
      4
    )} rad</strong> between one token and the next. Interpolation at ${fmt(s, 0)}× leaves it at ${fmt(
      fastPlain / s,
      4
    )}; the base change leaves it at <strong>${fmt(
      ntk.freqs[0],
      4
    )}</strong>, untouched. The slowest pair is compressed identically by both — ${slowPlain.toExponential(
      1
    )} becomes ${(slowPlain / s).toExponential(1)} either way. YaRN puts the reason in one parenthesis: <em>“the rotation describing the smallest distance needs to not be too small for the network to be able to detect it.”</em> On your own sentence the consequence is measurable: for the same nominal ${fmt(
      s,
      0
    )}×, interpolation moves some attention weight by <strong>${fmt(
      movedPi,
      3
    )}</strong> and the base change by <strong>${fmt(
      movedNtk,
      3
    )}</strong>. <span class="warn">That is a smaller <em>disturbance</em>, not a better answer.</span> These weights were never trained, so nothing here says the model reads better — only that less of what it already does has been rearranged. On a trained model that difference is what "no fine-tuning needed" means, and this app cannot reach it.`;

    // --- 3. out of bounds
    const dims = Number(width);
    const wide = dims === 128;
    boundBars8.node.style.display = wide ? "none" : "";
    boundBars128.node.style.display = wide ? "" : "none";
    const trainL = wide ? 2048 : L;
    const fPlain = wide ? freqsAt(B, dims) : plain.freqs;
    const fNtk = wide ? freqsAt(newBase(s, dims), dims) : ntk.freqs;
    const idx = wide ? SAMPLE_128 : Array.from({ length: PAIRS }, (_, i) => i);
    const rows = {};
    for (const i of idx) {
      const os = overshoot(s, i, dims);
      const turns = (trainL * fPlain[i]) / (2 * Math.PI);
      rows[`p${i}`] = {
        value: 1 / os,
        of: 1,
        text: `${fmt(os, 2)}× over · ${turns >= 0.01 ? fmt(turns, 2) : turns.toExponential(1)} turns`,
        tone: turns < 1 ? "alt" : "",
      };
    }
    (wide ? boundBars128 : boundBars8).update(rows);
    let over = 0;
    let never = 0;
    let both = 0;
    for (let i = 0; i < dims / 2; i++) {
      const os = overshoot(s, i, dims) > 1.0001;
      const nv = (2 * Math.PI) / fPlain[i] > trainL;
      if (os) over++;
      if (nv) never++;
      if (os && nv) both++;
    }
    const half = dims / 2;
    boundRead.update({
      over: `${over} of ${half}`,
      never: `${never} of ${half}`,
      both: `${both} of ${half}`,
    });
    boundNote.className = "note " + (both > 0 ? "warn" : "");
    boundNote.innerHTML = `Each bar is the fraction of the arc a pair reaches at position ${fmt(
      s * trainL,
      0
    )} that it had already seen by position ${trainL}; the number beside it is how far past that it goes, and how many full turns it completed while training. YaRN's sentence is <em>“some dimensions are slightly extrapolated to ‘out-of-bound’ values”</em>, and this is that sentence counted: <strong>${over} of ${half}</strong> pairs finish past their trained angle, the fastest by the full ${fmt(
      s,
      0
    )}×, and only the slowest lands exactly on the boundary. But overshoot is not one kind of thing. A pair that went round the circle ${fmt(
      (fPlain[0] * trainL) / (2 * Math.PI),
      1
    )} times during training has already been shown angles from every part of it, so ending past its largest one means little; a pair that never completed a single turn has a genuinely unexplored range. <strong>${never} of ${half}</strong> pairs never complete a turn, and <strong>${both} of ${half}</strong> do both — those are the dimensions the objection is really about, and their overshoot is bounded by about 2×, not ${fmt(
      s,
      0
    )}×. ${
      wide
        ? "These are the real numbers, at a 128-wide head with a 2,048-token training length."
        : "Switch the width to 128 for a real head's numbers; at this app's four pairs the picture is the same shape, coarser."
    } <span class="warn">The counts are this app's arithmetic from the definitions — YaRN gives none.</span>`;

    // --- 4. the dynamic variant
    const shipped = (l) => (l <= L ? B : B * Math.pow((alpha * l) / L - (alpha - 1), EXP));
    const yarnRule = (l) => B * Math.pow(Math.max(1, l / L), EXP);
    const xmax = alpha * L;
    const pts = [];
    const ref = [];
    for (let l = 1; l <= xmax; l++) {
      pts.push([l, shipped(l)]);
      ref.push([l, yarnRule(l)]);
    }
    const ymax = Math.max(shipped(xmax), yarnRule(xmax));
    dynCurve.update({
      points: pts,
      reference: ref,
      xRange: [0, xmax],
      yRange: [0, ymax],
      band: [0, L],
      mark: L,
      markLabel: "the trained length — below here, untouched",
    });
    // Dynamic scaling at a length inside the trained window is the identity, so this must be 0.
    const dynAtT = maxDiff(baseline, matrix(rope({ base: shipped(T), dims: DH })));
    const gotAlpha = alpha * alpha - alpha + 1;
    dynRead.update({
      asked: `${alpha}×`,
      got: `${fmt(gotAlpha, 0)}×`,
      short: dynAtT.toExponential(1),
      shortStatic: fmt(movedNtk, 3),
    });
    dynNote.className = "note";
    dynNote.innerHTML = `A static scale taxes every sequence with the compression the longest one needs. The variant that arrived one day later — from the other author, and the reason the record's date is a day out — recomputes the scale from the length actually being processed, so below the trained length the base is exactly ${B.toLocaleString()} and <strong>the model is the original, bit for bit</strong>: on your sentence the attention matrix differs from plain rotation by <strong>${dynAtT.toExponential(
      1
    )}</strong>, where the static setting above has already moved it by ${fmt(
      movedNtk,
      3
    )}. That is what graceful degradation buys, and it costs nothing. The solid line is the ramp that actually shipped in <code>transformers</code>, α·l′/L − (α−1), built so the bracket is exactly 1 at the trained length; the dashed line is the paper's plainer max(1, l′/L). Notice where they end: at ${fmt(
      xmax,
      0
    )} tokens you asked for <strong>${alpha}×</strong> and the shipped ramp is applying <strong>${fmt(
      gotAlpha,
      0
    )}×</strong>, because α²−α+1 is not α. Whether that is a bug or deliberate over-scaling — YaRN does say <em>“the scale value s has to be set higher than the expected scale”</em> — is not settled by anything published. One implementation trap comes with it, and people shipped the bug: <span class="warn">a key/value cache must store the vectors <em>before</em> rotation here</span>, because every token's rotation changes the moment the scale does.`;
  }

  return { update: render, unmount: () => {} };
}
