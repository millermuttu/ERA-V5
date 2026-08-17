// Concept 9 — linear attention, the kernel regrouping.
// Built from docs/research/linear-attention.md. Three corrections the research forced: the paper
// claims "similar", not parity, and its own numbers are worse on two of three tasks; the 4,000x
// headline is against a naive recompute baseline (about 56x against a real KV cache); and the
// lesson's 140 = 140 is the numerator only — the denominator regroups too, and dropping it is the
// most common way to get this wrong.
import { el, slider, toggle, svg } from "../lib/dom.js";
import { readout } from "../views/bars.js";
import { dot, fmt } from "../model/ops.js";
import { forward, CONFIG, DH } from "../model/transformer.js";
import { softmaxMixer, stateMixer, elu1 } from "../model/mixers.js";
import { softmax } from "../model/ops.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose } from "./chrome.js";

export function linearAttentionCard(root, m) {
  let q = 2;
  let useSoftmax = false;
  let step = null;
  let badPhi = false;

  root.appendChild(
    prose({
      problem:
        "Sparse patterns and windows reduce how many keys a query reads, but the query still reads individual keys and the cache still holds an entry per token. The deeper obstacle is softmax itself: its denominator sums over every score, so the weight of one key depends on all the others, and nothing can be summarised before the query arrives.",
      mechanism:
        "Replace the exponential with a feature map φ, so the similarity becomes φ(q)·φ(k). Then the sum can be reassociated: instead of (φ(Q)φ(K)ᵀ)V, which builds an N×N matrix, compute φ(Q)(φ(K)ᵀV), which never does. The bracketed part does not involve the query, so it can be accumulated as the tokens arrive into a fixed d×d state — and a second running sum for the denominator. At generation time this is literally an RNN: one state in, one state out, constant memory per step. The feature map is elu(x)+1, chosen because it is non-negative and, unlike relu, has no zero-gradient region — not because it resembles the exponential.",
    })
  );

  // -------------------------------------------------------------- two routes
  const qSlider = slider({
    label: "query q",
    min: -3,
    max: 5,
    step: 0.5,
    value: 2,
    oninput: (v) => ((q = v), render()),
  });
  const smToggle = toggle({ label: "softmax", value: false, onchange: (v) => ((useSoftmax = v), render()) });
  const routeTable = el("div", { class: "routes" });
  const routeRead = readout([
    { key: "diff", label: "difference between the routes" },
    { key: "weights", label: "where the weight lands" },
  ]);
  const routeNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the same sum, grouped two ways" }),
      el("div", { class: "ctrls" }, [qSlider.node, smToggle]),
      routeTable,
      routeRead.node,
      routeNote,
    ])
  );

  // ------------------------------------------------------------- the state
  const stepSlider = slider({
    label: "tokens seen",
    min: 1,
    max: 16,
    value: 16,
    oninput: (v) => ((step = v), render()),
  });
  const stateWrap = el("div", { class: "statewrap" });
  const sizeRead = readout([
    { key: "state", label: "numbers in the state" },
    { key: "cache", label: "numbers a softmax cache would hold" },
    { key: "cross", label: "they crossed at" },
  ]);
  const stateNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the state does not grow" }),
      el("div", { class: "ctrls" }, [stepSlider.node]),
      stateWrap,
      sizeRead.node,
      stateNote,
    ])
  );

  // ---------------------------------------------------- what the answer costs
  const compareRead = readout([
    { key: "agree", label: "top word each one attends to" },
    { key: "drift", label: "how far the outputs differ" },
    { key: "ops", label: "multiplies at this length" },
  ]);
  const compareNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "and what the compression costs on your sentence" }),
      compareRead.node,
      compareNote,
    ])
  );

  // ------------------------------------------------ why the feature map must be positive
  const phiToggle = toggle({
    label: "use a feature map that can go negative",
    value: false,
    onchange: (v) => ((badPhi = v), render()),
  });
  const phiRead = readout([
    { key: "neg", label: "tokens with a negative denominator" },
    { key: "worst", label: "worst denominator" },
  ]);
  const phiNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "why the feature map has to be non-negative" }),
      el("div", { class: "ctrls" }, [phiToggle]),
      phiRead.node,
      phiNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Sequence work becomes linear, and generation becomes constant-memory per token — the model is literally an RNN at inference",
        "The key/value cache stops existing: one fixed state replaces the growing history, whatever the length",
        "The regrouping is exact arithmetic, not an approximation — given the feature map, both routes agree to the last decimal",
      ],
      givesUp: [
        "Exact token-level lookup. The past is compressed into one object and different memories interfere",
        "It does not approximate softmax and the paper says so — the exponential kernel's feature map is infinite-dimensional, making the linearisation infeasible",
        "Quality: the paper reports similar rather than equal, and its own numbers are worse on two of three tasks — 0.644 against 0.621 bits per dimension on MNIST, and 8.08 against 5.12 phone error rate on speech, where the authors write that softmax outperforms significantly",
        "The 4,000× headline is against recomputing everything from scratch; against a real key/value cache the same table gives about 56×",
      ],
      chooseWhen:
        "Very long sequences where a growing cache is fatal and the task tolerates a lossy summary — and note there is no language-modelling experiment in the paper at all. Everything after this on the timeline is repairing what removing softmax broke.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The notes the model keeps while writing stop growing — one fixed-size summary replaces a list that gets longer with every word",
        "Producing each new word costs the same whether it is the tenth or the ten-thousandth",
        "Regrouping the sum is exact arithmetic, not a guess: both ways of adding it up give the identical answer",
      ],
      cons: [
        "The summary is lossy — everything the model read is squeezed into one small grid, and memories smudge into each other",
        "It is not an imitation of the original mechanism, and the authors are explicit that imitating it exactly is not possible this way",
        "On the paper's own tests it was clearly worse at two of the three tasks, and the huge speed figure is measured against a deliberately wasteful baseline",
      ],
      verdict:
        "Move one pair of brackets and the growing memory disappears. What you buy is a model that can write forever at a fixed cost; what you pay is that everything it read is now a single blurred summary rather than a list it can look things up in. The next two concepts on this timeline exist to make that summary less blurred.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    qSlider.set(q);
    if (step === null || step > T) step = T;
    stepSlider.input.max = T;
    stepSlider.set(Math.min(step, T));

    // --- 1. the two routes, on the lesson's numbers
    const keys = [0.5, 1.0, 1.5];
    const vals = [10, 20, 30];
    const raw = keys.map((k) => q * k);
    const numDirect = raw.reduce((s, w, i) => s + w * vals[i], 0);
    const denDirect = raw.reduce((s, w) => s + w, 0);
    const S = keys.reduce((s, k, i) => s + k * vals[i], 0);
    const Z = keys.reduce((s, k) => s + k, 0);
    const numRegrouped = q * S;
    const denRegrouped = q * Z;
    const linWeights = raw.map((w) => w / denDirect);
    const smWeights = softmax(raw);
    const smOut = smWeights.reduce((s, w, i) => s + w * vals[i], 0);

    const row = (label, a, b) =>
      el("div", { class: "rrow" }, [
        el("span", { class: "rlab", text: label }),
        el("span", { class: "rval", text: a }),
        el("span", { class: "rval", text: b }),
      ]);

    routeTable.replaceChildren(
      el("div", { class: "rrow rhead" }, [
        el("span", { class: "rlab", text: "" }),
        el("span", { class: "rval", text: useSoftmax ? "softmax route" : "direct route" }),
        el("span", { class: "rval", text: "regrouped route" }),
      ]),
      row(
        "numerator",
        useSoftmax ? "—" : `${raw.map((w, i) => `${fmt(w, 1)}×${vals[i]}`).join(" + ")} = ${fmt(numDirect, 1)}`,
        useSoftmax ? "no state exists" : `S = ${fmt(S, 1)}, q×S = ${fmt(numRegrouped, 1)}`
      ),
      row(
        "denominator",
        useSoftmax ? "sum of exponentials" : `${raw.map((w) => fmt(w, 1)).join(" + ")} = ${fmt(denDirect, 1)}`,
        useSoftmax ? "—" : `Z = ${fmt(Z, 1)}, q×Z = ${fmt(denRegrouped, 1)}`
      ),
      row(
        "output",
        useSoftmax ? fmt(smOut, 2) : fmt(numDirect / denDirect, 2),
        useSoftmax ? "—" : fmt(numRegrouped / denRegrouped, 2)
      )
    );

    const diff = Math.abs(numDirect / denDirect - numRegrouped / denRegrouped);
    routeRead.update({
      diff: useSoftmax ? "the regrouping does not exist" : diff.toFixed(6),
      weights: useSoftmax
        ? smWeights.map((w) => fmt(w, 3)).join("  ")
        : linWeights.map((w) => fmt(w, 3)).join("  "),
    });
    routeNote.className = "note " + (useSoftmax ? "warn" : "good");
    routeNote.innerHTML = useSoftmax
      ? `With softmax there is no S to build: the denominator is a sum of exponentials over every score, so the weight of one key cannot be known until all of them are. The regrouped column is blank rather than wrong — the operation is undefined, not inaccurate. Notice where the weight goes: softmax gives the last key ${fmt(smWeights[2] * 100, 1)}% against the linear route's ${fmt(linWeights[2] * 100, 1)}%. That sharpening is what is being given up.`
      : `Both routes give ${fmt(numDirect / denDirect, 2)}, differing by ${diff.toFixed(6)}. Drag the query: they agree at every value, because this is arithmetic rather than approximation. Note the denominator row — the lesson's 140 = 140 is only the numerator, and Z regroups exactly the same way. Dropping it is the most common way to implement this wrongly.`;

    // --- 2. the state
    const lin = forward(tokens, { mixer: stateMixer({ write: "add" }) });
    const h = lin.trace[0].heads[0];
    const snap = h.snapshots[Math.min(step, T) - 1];
    const maxAbs = Math.max(...snap.flatMap((r) => Array.from(r, Math.abs)), 1e-9);
    const CELL = 20;
    stateWrap.replaceChildren(
      svg(
        "svg",
        {
          viewBox: `0 0 ${DH * CELL + 120} ${DH * CELL + 24}`,
          width: DH * CELL + 120,
          height: DH * CELL + 24,
          class: "diagram",
          role: "img",
          "aria-label": "the state matrix after the chosen number of tokens",
        },
        [
          ...snap.flatMap((rr, a) =>
            Array.from(rr, (x, b) =>
              svg("rect", {
                x: b * CELL,
                y: 18 + a * CELL,
                width: CELL - 2,
                height: CELL - 2,
                rx: 2,
                fill:
                  x >= 0
                    ? `rgba(79,197,140,${(0.08 + (Math.abs(x) / maxAbs) * 0.85).toFixed(3)})`
                    : `rgba(224,105,61,${(0.08 + (Math.abs(x) / maxAbs) * 0.85).toFixed(3)})`,
              })
            )
          ),
          svg("text", { x: 0, y: 12, class: "gridlab" }, [`S after ${Math.min(step, T)} token${step > 1 ? "s" : ""}`]),
          svg("text", { x: DH * CELL + 12, y: 34, class: "gridlab" }, [`${DH} × ${DH}`]),
          svg("text", { x: DH * CELL + 12, y: 52, class: "gridlab" }, ["always"]),
        ]
      )
    );

    const stateNumbers = DH * DH + DH; // S plus the denominator vector Z
    const cachePerToken = 2 * DH;
    const crossAt = Math.ceil(stateNumbers / cachePerToken);
    sizeRead.update({
      state: String(stateNumbers),
      cache: String(cachePerToken * Math.min(step, T)),
      cross: `${crossAt} tokens`,
    });
    stateNote.innerHTML = `Step the slider and watch each token add its own rank-one pattern to the same grid — the numbers change, the grid does not grow. The state is ${DH}×${DH} plus a ${DH}-number denominator: <strong>${stateNumbers} numbers after one token and after a million</strong>. A softmax cache stores ${cachePerToken} numbers per token, so it passes the fixed state at ${crossAt} tokens and keeps going. Being honest about scale: at your sentence's length the two are comparable, and the compute crossover for this model is around ${2 * DH - 1} tokens. This mechanism is not for sentences.`;

    // --- 3. what it costs on this sentence
    const sm = forward(tokens, {});
    const smH = sm.trace[0].heads[0];
    const qIdx = T - 1;
    const smTop = smH.weights[qIdx].indexOf(Math.max(...smH.weights[qIdx]));
    let drift = 0;
    for (let d = 0; d < DH; d++) drift += (smH.out[qIdx][d] - h.out[qIdx][d]) ** 2;
    drift = Math.sqrt(drift);
    // The linear model has no weights to inspect, so ask which value it leans on most.
    let linTop = 0;
    let bestSim = -Infinity;
    for (let j = 0; j <= qIdx; j++) {
      const sim = dot(Array.from(h.Q[qIdx], elu1), Array.from(h.K[j], elu1));
      if (sim > bestSim) {
        bestSim = sim;
        linTop = j;
      }
    }
    compareRead.update({
      agree: `softmax “${tokens[smTop].word}” · linear “${tokens[linTop].word}”`,
      drift: fmt(drift, 3),
      ops: `${(2 * DH * DH * T).toLocaleString()} vs ${(T * (T + 1) * DH).toLocaleString()}`,
    });
    compareNote.textContent = `Run the same sentence through both mixers and the outputs differ by ${fmt(drift, 3)} at the last token. The two do not agree on what to look at either. This is not a quality measurement — these weights are untrained and neither answer means anything — it is a demonstration that removing softmax changes the mechanism rather than merely speeding it up. The paper is careful about this too: it reports "similar" performance, and on speech recognition its own numbers are 8.08 against softmax's 5.12.`;

    // --- 4. non-negativity
    const bad = forward(tokens, { mixer: stateMixer({ write: "add", phi: (x) => x }) });
    const dens = badPhi ? bad.trace[0].heads[0].denominators : h.denominators;
    const negs = dens.filter((z) => z < 0).length;
    const worst = dens.reduce((a, b) => (b < a ? b : a), Infinity);
    phiRead.update({ neg: `${negs} of ${dens.length}`, worst: fmt(worst, 3) });
    phiNote.className = "note " + (negs > 0 ? "warn" : "good");
    phiNote.textContent = negs
      ? `With a feature map that can go negative, ${negs} of ${dens.length} tokens end up with a negative denominator — the worst is ${fmt(worst, 3)}. Dividing by that flips the sign of the output: the "attention weights" are no longer weights at all, and nothing is a weighted average of anything. This is the one hard requirement the paper places on the similarity function, and it is why elu(x)+1 is used rather than the identity.`
      : `Every denominator is positive, so every output is a genuine weighted average of the values. That is the whole reason the feature map must be non-negative — and the paper's choice of elu(x)+1 over relu is about the gradient, not about the sign: relu is non-negative too, but it has a region where it teaches the model nothing.`;
  }

  return { update: render, unmount: () => {} };
}
