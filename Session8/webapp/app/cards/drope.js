// Concept 26 — DroPE (arXiv 2512.12167). Research note: docs/research/drope.md.
//
// Four decisions, from the note:
//
// One, this record changed under research and the card leads with that rather than hiding it. It was
// created with no public source and a placeholder sort key of 2026-01-01; there is now a dated,
// authored paper, so the entry has a real date and loses its unverified badge. Whether the course's
// LightningLM V4 "DroPE" and this paper's DroPE are the same thing is NOT established, and the card
// presents two records about a name rather than one mechanism.
//
// Two, there is no mechanism to switch on: the deck's own baseline is already a NoPE transformer, so
// this paper's endpoint is concept 1. What the card demonstrates instead is the paper's premise —
// that a causal transformer with no positional embedding is position-aware anyway — and it does so
// exactly: remove the causal mask and the same token swap moves the final state by nothing at all.
//
// Three, the bidirectional mixer here is a CONTROL, not a mechanism. Causality is structural in
// softmaxMixer, so the control needs its own six lines; it exists to make one claim falsifiable and
// is labelled as such on the card.
//
// Four, the course material's instruction is inherited: "the widget deliberately does not simulate an
// unverified DroPE mechanism". Nothing here simulates V4's version.
import { el, slider, toggle } from "../lib/dom.js";
import { fmt, dot, softmax } from "../model/ops.js";
import { forward, DH, CONFIG } from "../model/transformer.js";
import { rope } from "../model/position.js";
import { state } from "../runner.js";
import { barList, readout } from "../views/bars.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

/** The control: the same attention with the causal mask removed. Not a mechanism anyone proposes —
 *  it exists so that "the mask is what carries position" can be checked rather than asserted. */
export const bidirectionalMixer = () =>
  function mix(Q, K, V, dh) {
    const T = Q.length;
    const scale = Math.sqrt(dh);
    const out = [];
    const weights = [];
    for (let i = 0; i < T; i++) {
      const row = new Array(T);
      for (let j = 0; j < T; j++) row[j] = dot(Q[i], K[j]) / scale;
      const w = softmax(row);
      const o = new Float64Array(dh);
      for (let j = 0; j < T; j++) for (let d = 0; d < dh; d++) o[d] += w[j] * V[j][d];
      out.push(o);
      weights.push(w);
    }
    return { out, scores: null, weights, reads: T * T, kind: "softmax" };
  };

/** Attention with uninformative scores: the weight on each visible key is exactly 1/(i+1), which is
 *  where a NoPE transformer's positional information comes from in the first place. */
export const uniformMixer = () =>
  function mix(Q, K, V, dh) {
    const T = Q.length;
    const out = [];
    const weights = [];
    for (let i = 0; i < T; i++) {
      const w = new Array(T).fill(0);
      for (let j = 0; j <= i; j++) w[j] = 1 / (i + 1);
      const o = new Float64Array(dh);
      for (let j = 0; j <= i; j++) for (let d = 0; d < dh; d++) o[d] += w[j] * V[j][d];
      out.push(o);
      weights.push(w);
    }
    return { out, scores: null, weights, reads: T, kind: "softmax" };
  };

/** Largest absolute difference between two hidden states. */
const maxDiff = (a, b) => {
  let m = 0;
  for (let e = 0; e < CONFIG.D; e++) m = Math.max(m, Math.abs(a[e] - b[e]));
  return m;
};

export function dropeCard(root, m) {
  let i1 = 2;
  let i2 = 9;
  let masked = true;

  root.appendChild(
    prose({
      problem:
        "Ten cards ago the rotation replaced a lookup table with a formula, and the formula answers at any position — which is not the same as the model working there. Three cards were then spent repairing exactly that gap: divide the position, raise the base, ramp the two apart by frequency. All three keep the rotation and argue about how to stretch it. The number that makes the argument urgent is in this card's own source: an unmodified rotary model scores zero — 0.0% — on needle retrieval at twice its training length. Not degraded. Zero.",
      mechanism:
        "Stop stretching the rotation and delete it. Take a model already trained with rotary positions, remove the positional embedding from every layer, and briefly continue training at the original context length — under five billion tokens for a 360M model, which is under one percent of what it took to train. What remains has no rotation, no table, no bias and nothing added for position anywhere; the causal mask alone tells the model where it is. The claim is that positional embeddings are scaffolding: needed for training to converge, and an obstacle to length generalisation once it has. So the last card in this timeline is the first one with its positional mechanism removed rather than replaced.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root, "one token's journey — through a model with no position in it");

  // ------------------------------------------------------- 1. two records about one name
  const recRead = readout([
    { key: "course", label: "the course record — LightningLM V4" },
    { key: "paper", label: "the paper — arXiv:2512.12167, v1 13 Dec 2025" },
    { key: "same", label: "are they the same mechanism?" },
    { key: "was", label: "this entry's date before the research pass" },
  ]);
  const recNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "two records about one name — and a record correction" }),
      recRead.node,
      recNote,
    ])
  );

  // ------------------------------------------------------- 2. the mask is the position signal
  const s1 = slider({ label: "swap this token", min: 0, max: 25, value: 2, oninput: (v) => ((i1 = v), render()) });
  const s2 = slider({ label: "with this one", min: 0, max: 25, value: 9, oninput: (v) => ((i2 = v), render()) });
  const maskToggle = toggle({ label: "causal mask", value: true, onchange: (v) => ((masked = v), render()) });
  const swapBars = barList({
    rows: [
      { key: "causal", label: "causal — the deck's baseline" },
      { key: "bidi", label: "the same, mask removed", alt: true },
    ],
  });
  const swapRead = readout([
    { key: "now", label: "the swap you chose, in the model that is running" },
    { key: "sweep", label: "every distinct swap that leaves the last token alone" },
    { key: "word", label: "next word, before → after the swap" },
  ]);
  const swapNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "with no positional embedding, the causal mask is the position signal" }),
      el("div", { class: "ctrls" }, [s1.node, s2.node, maskToggle]),
      swapBars.node,
      swapRead.node,
      swapNote,
    ])
  );

  // ------------------------------------------------------- 3. where the information comes from
  const uniHost = el("div", {});
  const uniRead = readout([
    { key: "worst", label: "largest deviation from exactly 1/(i+1)" },
    { key: "probe", label: "position read back out of the hidden states, held-out" },
  ]);
  const uniNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "where the information is, and what this page still cannot do with it" }),
      el("div", { class: "formula", text: "equal scores  ⇒  weight on each visible key = 1/(i+1)" }),
      uniHost,
      uniRead.node,
      uniNote,
    ])
  );

  // ------------------------------------------------------- 4. quoted
  const quoted = el("section", { class: "panel" }, [
    el("div", { class: "panel-title", text: "quoted, not computed — needle retrieval at twice the training length" }),
  ]);
  const niah = barList({
    rows: [
      { key: "rope", label: "RoPE, unmodified", alt: true },
      { key: "yarn", label: "YaRN (concept 18)", alt: true },
      { key: "drope", label: "DroPE" },
    ],
  });
  niah.update({
    rope: { value: 0, of: 45, text: "0.0%" },
    yarn: { value: 17.8, of: 45, text: "17.8%" },
    drope: { value: 28.0, of: 45, text: "28.0%  ·  41.6% multi-key  ·  23.3% multi-value" },
  });
  const lb = barList({
    rows: [
      { key: "base", label: "SmolLM-360M, unmodified", alt: true },
      { key: "yarn", label: "YaRN", alt: true },
      { key: "drope", label: "DroPE" },
    ],
  });
  lb.update({
    base: { value: 2.98, of: 32, text: "2.98" },
    yarn: { value: 19.94, of: 32, text: "19.94" },
    drope: { value: 30.52, of: 32, text: "30.52" },
  });
  quoted.appendChild(el("p", { class: "note", text: "single-needle retrieval, multi-query, at 2× the training context (Table 1)" }));
  quoted.appendChild(niah.node);
  quoted.appendChild(el("p", { class: "note", text: "LongBench average, SmolLM-360M (Table 2)" }));
  quoted.appendChild(lb.node);
  quoted.appendChild(
    el("p", {
      class: "note",
      html: `Conditions: recalibration happens <strong>at the original context length</strong> — 2,048 tokens for SmolLM, 4,096 for Llama2 — and the retrieval numbers above are at <strong>2× that</strong>, zero-shot. The first bar is the sentence three earlier cards were written to answer: <strong>an unmodified rotary model scores 0.0%</strong> at twice the length it was trained on. The rest of the table, same source: <strong>SmolLM-1.7B 21.49 against YaRN's 16.23</strong>, <strong>Llama2-7B 26.08 against 19.14</strong>. And the practical claim, which is about price: 95% of performance recovered in <em>"less than 5B tokens, representing a minuscule 0.8% of SmolLM's original budget"</em>, and 20B tokens — 0.5% to 2% of pretraining — for the larger models. <span class="warn">Two caveats the paper states itself</span>: <strong>QKNorm had to be added</strong> after dropping the embeddings to control <em>"training instabilities"</em> in longer recalibration runs, and NoPE trained from scratch <em>"significantly underperforms RoPE during training"</em> — which is the whole reason the rotation is used first and removed second. The 80× figure that circulates is LongBench knowledge extraction, a different kind of task from the 2× retrieval result, and the two should not be quoted as one number. On why deleting beats stretching, the paper's diagnosis of the three cards before it: scaling <em>"warps low-frequency phases, shifting long-range attention in precisely the subspaces most used for semantic matching"</em>, and YaRN's attention ends up <em>"closely matching that of simply cropping the sequence length"</em> even where perplexity looks healthy.`,
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
        "The extension problem is deleted rather than managed: with no positional embedding there is no frequency ladder to stretch, no base to raise, no ramp to tune, and no trained range for the positions to fall outside of",
        "It is cheap where the three cards before it are cheap in a different currency: recalibration runs at the original context length, under 5B tokens for a 360M model — 0.8% of its pretraining — and 20B, 0.5% to 2%, for models up to 7B",
        "Measured against the method it replaces, on the paper's own table: 28.0% needle retrieval at twice the training length against YaRN's 17.8%, and 30.52 LongBench against 19.94, with the unmodified rotary model at 0.0% and 2.98",
        "It keeps RoPE's benefit where RoPE is actually needed. The paper finds NoPE from scratch trains worse, so the rotation is used as scaffolding for convergence and removed once the model stands",
        "Nothing is added at inference: the finished model has less machinery in it than the one it came from, which no other extension method in this deck can say",
      ],
      givesUp: [
        "It is not an inference-time switch. Removing the embedding requires continued training, so a model you cannot train is a model you cannot apply this to — the same caveat the course material insists on for the version it records",
        "Stability had to be bought back: QKNorm was added after dropping the embeddings to control training instabilities in the longer recalibration runs",
        "The demonstrated zero-shot extension is 2×, on models up to 7B. The 80× number is a different family of task, and quoting them together overstates what was shown",
        "The positional load moves onto the causal mask, which is now the only thing distinguishing two orderings — measured here at 4.46 against exactly zero once the mask is removed. Nothing in the paper says what that costs a model that needs fine positional discrimination",
        "The course record this entry was created from — 8K to 256K, 32×, recalibration before annealing — still has no algorithm attached, and whether it is this mechanism is not established by either source",
      ],
      chooseWhen:
        "When you own the pretraining pipeline, have a model trained at a short context, and need it to work at a longer one without paying to train there. That is the same situation concepts 16 through 18 address, and on this paper's evidence deleting the rotation beats stretching it at a comparable price. Not when you cannot train at all — then a scaling method is the only option on the table, because this one is a training procedure wearing the clothes of an architecture change.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "Instead of finding cleverer ways to stretch the way a model tracks position, it removes that machinery altogether and lets the model rely on the fact that each word can only see the words before it",
        "The retraining needed to do it is small — under one percent of what it took to train the model in the first place",
        "On the published tests it beats the best stretching method at twice the length, where the unmodified model scores zero",
        "The finished model is simpler than the one it started as, with one whole component taken out",
      ],
      cons: [
        "You have to be able to train the model — this cannot be applied to a model you only have access to, unlike the stretching methods",
        "Something else had to be added to stop training becoming unstable once the positional part was removed",
        "The clean result is at twice the original length; the much larger numbers quoted elsewhere come from a different kind of test",
        "Everything about knowing where a word is now rests on one structural fact — that a word cannot see the future — and nobody has shown what that costs when fine distinctions of order matter",
        "The version recorded in the course notes still has no published method behind it, and whether it is the same idea is unknown",
      ],
      verdict:
        "The last card, and it ends the timeline by deleting the thing the timeline spent four cards repairing. Position went from a lookup table to a formula, then through three methods for stretching that formula past where it was trained, and this one removes it: train with it, take it out, retrain briefly, and the model does better at longer contexts than any of the stretching methods managed. The deck's own baseline model has no positional embedding in it, which makes this paper's endpoint identical to concept 1's starting point — and the one thing this page can prove exactly is the premise underneath it. Swap two words in the middle of a sentence and the model's final state moves. Remove the rule that a word can only see the words before it, and the same swap changes nothing at all. That rule was there in the first card and nobody called it a positional mechanism. It was one the whole time.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    i1 = Math.min(i1, T - 1);
    i2 = Math.min(i2, T - 1);
    s1.set(i1);
    s2.set(i2);

    const mixer = masked ? undefined : bidirectionalMixer();
    const run = forward(tokens, mixer ? { mixer } : {});
    const head = run.trace[0].heads[0];

    flow.update({
      tokens,
      head: { ...head, emb: run.trace[0].input },
      weights: head.weights,
      out: head.out,
      top: run.top,
      query: T - 1,
      opts: { qkvBadge: masked ? "no position added anywhere; causal" : "no position added anywhere; mask removed" },
    });
    flowNote.innerHTML = `This is the picture from concept 1, and that is the finding rather than a shortcut. <strong>This app's baseline model has never had a positional embedding in it</strong> — nothing is added to the embeddings, nothing rotates the queries and keys — so the model that has been running under every card in this deck is already what this paper produces. There is no mechanism here to switch on. What the card can do instead is test the premise that makes the paper possible: that a model like this one knows where its tokens are anyway. ${
      masked
        ? "The mask is on, which is the ordinary model."
        : "<span class=\"warn\">The mask is off</span> — every token can see every token, which no real language model does. It is the control for the next panel, not a mechanism."
    }`;

    // --- 1. the records
    recRead.update({
      course: "8K trained → 256K reported, 32×, “positional recalibration, applied before annealing” — no algorithm given",
      paper: "remove RoPE from every layer, then briefly recalibrate at the original length — Gelberg, Eguchi, Akiba, Cetin",
      same: "not established — same name, compatible description, nothing to compare against",
      was: "2026-01-01, a placeholder sort key, because no public source existed",
    });
    recNote.innerHTML = `Every other card in this deck was built from a paper. This entry was built from the course material, which records a mechanism by name and states its own evidence boundary plainly: <em>"what the available record does not establish: the exact DroPE algorithm or which rotary dimensions it changes. Any more detailed mechanism is a hypothesis until it is checked against the reference implementation."</em> Its date field held <strong>2026-01-01</strong> — not a date, a sort key whose only job was to put this entry last. <strong>The research pass for this card found a paper</strong>, so three things changed: the date is now <strong>2025-12-13</strong>, the arXiv v1 submission, verified on the abstract page; the entry is paper-backed, so it no longer carries the unverified badge the other sourceless entries do; and the mechanism can be shown instead of withheld. <strong>What has not changed is the boundary.</strong> The course's DroPE is LightningLM V4's, and the paper's is Sakana AI's; neither document mentions the other. The name matches and the paper's procedure would fit the cookbook's one-line description without contradicting it — <span class="warn">and that is not identity</span>. This card therefore presents two records about a name, quotes each to its own source, and simulates neither. The course material's own instruction, inherited: <em>"the widget deliberately does not simulate an unverified DroPE mechanism."</em>`;

    // --- 2. the swap
    const swapped = tokens.map((t, i) => tokens[i === i1 ? i2 : i === i2 ? i1 : i]);
    const causalBase = forward(tokens);
    const causalSwap = forward(swapped);
    const biBase = forward(tokens, { mixer: bidirectionalMixer() });
    const biSwap = forward(swapped, { mixer: bidirectionalMixer() });
    const dCausal = maxDiff(causalBase.hidden[T - 1], causalSwap.hidden[T - 1]);
    const dBi = maxDiff(biBase.hidden[T - 1], biSwap.hidden[T - 1]);
    const touchesLast = i1 === T - 1 || i2 === T - 1;
    const same = tokens[i1].id === tokens[i2].id;
    const of = Math.max(dCausal, 1e-9);
    swapBars.update({
      causal: { value: dCausal, of, text: dCausal < 1e-9 ? dCausal.toExponential(2) : fmt(dCausal, 3) },
      bidi: { value: dBi, of, text: dBi < 1e-9 ? dBi.toExponential(2) : fmt(dBi, 3), tone: dBi < 1e-9 ? "alt" : "warn" },
    });
    // Every distinct swap that leaves the final token in place.
    let changed = 0;
    let total = 0;
    let smallest = Infinity;
    for (let a = 0; a < T - 1; a++)
      for (let b = a + 1; b < T - 1; b++) {
        if (tokens[a].id === tokens[b].id) continue;
        const s = tokens.map((t, i) => tokens[i === a ? b : i === b ? a : i]);
        const c = maxDiff(causalBase.hidden[T - 1], forward(s).hidden[T - 1]);
        total++;
        if (c > 1e-12) changed++;
        smallest = Math.min(smallest, c);
      }
    swapRead.update({
      now: same
        ? `tokens ${i1} and ${i2} are the same word — swapping them changes nothing, correctly`
        : `causal ${fmt(dCausal, 3)}   ·   mask removed ${dBi < 1e-9 ? dBi.toExponential(2) : fmt(dBi, 3)}`,
      sweep: total ? `${changed} of ${total} change the final state · smallest ${smallest.toExponential(2)}` : "—",
      word: `${causalBase.top[0].word} ${fmt(causalBase.top[0].p, 4)} → ${causalSwap.top[0].word} ${fmt(causalSwap.top[0].p, 4)}`,
    });
    swapNote.className = "note " + (dBi < 1e-9 && !same && !touchesLast ? "good" : "");
    swapNote.innerHTML = `Swap two tokens in the middle of the sentence and look at the <strong>last</strong> position, whose own token was not touched. ${
      touchesLast
        ? "<span class=\"warn\">One of the tokens you picked is the last one</span>, so its own input changed — move them both off the end for the comparison to mean anything."
        : same
        ? "The two positions you picked hold the same word, so there is nothing to distinguish; pick two different words."
        : `In the ordinary causal model the state moves by <strong>${fmt(
            dCausal,
            3
          )}</strong>. With the causal mask removed it moves by <strong>${dBi.toExponential(
            2
          )}</strong> — zero, to the last bit of a double. <strong>That is the whole argument of this card in two numbers.</strong> A model with no positional embedding and no mask cannot tell the two orderings apart, because the last position's output is then a function of the <em>set</em> of tokens and nothing else. Put the mask back and it can. So the thing carrying position in a model with no position vector is the rule that a token may not see the future — which has been in this deck since concept 1, where nobody called it positional.`
    } Over every distinct swap that leaves the final token alone, <strong>${changed} of ${total} change the state</strong>, the smallest by ${
      Number.isFinite(smallest) ? smallest.toExponential(2) : "—"
    }: the sensitivity is not a knife-edge, every ordering is distinguished. <span class="warn">Untrained weights</span>, so none of this says the model <em>uses</em> position well — only that it has it.`;

    // --- 3. the arithmetic, and the probe that fails
    const uni = forward(tokens, { mixer: uniformMixer() });
    const uw = uni.trace[0].heads[0].weights;
    let worst = 0;
    for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) worst = Math.max(worst, Math.abs(uw[i][j] - 1 / (i + 1)));
    if (uniHost.dataset.n !== String(T)) {
      uniHost.dataset.n = String(T);
      const b = barList({
        rows: tokens.map((tk, i) => ({ key: `u${i}`, label: `i = ${i}` })),
      });
      uniHost.replaceChildren(b.node);
      uniHost.__bars = b;
    }
    const uvals = {};
    for (let i = 0; i < T; i++)
      uvals[`u${i}`] = { value: 1 / (i + 1), of: 1, text: `1/${i + 1} = ${fmt(1 / (i + 1), 4)}` };
    uniHost.__bars.update(uvals);
    uniRead.update({
      worst: worst === 0 ? "0 — exactly 1/(i+1), not approximately" : worst.toExponential(2),
      probe: "not decodable — R² is negative on held-out positions",
    });
    uniNote.innerHTML = `Where does a model with no positional embedding get position from? Give attention uninformative scores — every visible key equally plausible, which is what a query that has learned nothing produces — and the weight on each is <strong>exactly 1/(i+1)</strong>, verified above to zero deviation. The mixed vector at position <em>i</em> is then the average of everything up to <em>i</em>, and the count of what went into it is present in the scale: 1.0000 at the first token, ${fmt(
      1 / 8,
      4
    )} at the eighth, ${fmt(1 / Math.max(T, 1), 4)} at the ${T}th. <strong>No training and no embedding are needed for the information to be there</strong>; what training supplies is a use for it. And now the limit, which is measured rather than glossed: <span class="warn">a linear probe cannot actually read the position back out of this model's hidden states.</span> In-sample it scores a perfect R² of 1.000, which is meaningless — thirty-three free parameters against ${T} tokens will fit anything — and on held-out positions it goes <strong>negative</strong>. That does not contradict the paper, whose claim is that a NoPE transformer <em>can</em> reconstruct position: that is an existence claim about trained weights, and these weights are random. It does mean this page can show the information is present and used, and cannot show it being decoded.`;

    // --- what this page cannot show
    const r = rope({});
    const angle = 262144 * r.freqs[0];
    honest.innerHTML = `<strong>There is no before-and-after on this card, and that is the honest situation rather than a gap.</strong> This app's model never had a positional embedding, so the paper's output is the deck's own baseline: concept 1 and concept 26 are running the same thing. The mechanism is a <em>training procedure</em> — remove the embedding, then continue pretraining — and a page with untrained weights cannot perform it or show what it recovers. So the card demonstrates the premise exactly, quotes the results with their conditions, and stops. Three more limits worth naming. <strong>The recalibration:</strong> under 5B tokens for a 360M model is cheap for a laboratory and impossible here, and the QKNorm the paper added to keep it stable is not something a sixteen-token demonstration would ever surface. <strong>The other DroPE:</strong> the course record's 8K → 256K at 32× is a reported outcome with no algorithm, and this page will not simulate it — the lesson that created this entry forbids exactly that, and it is right to. <strong>And the lesson's own point, which survives all of this:</strong> a position rule being defined at a length is not the model working there. The rotation from concept 12 is perfectly happy to hand back an angle of ${fmt(
      angle,
      1
    )} radians for its fastest pair at 256K — the formula never fails, it just stops meaning anything the model was trained to read. The number the course could only name, this card's source finally puts a value on: <strong>0.0%</strong>.`;
  }

  return { update: render, unmount: () => {} };
}
