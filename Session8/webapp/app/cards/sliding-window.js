// Concept 8 — sliding window attention with global tokens (Longformer).
// Built from docs/research/sliding-window.md. The research corrected three things: the paper's
// global attention is never used in its autoregressive model, the ℓ×w receptive-field figure is
// the bidirectional case (causal gets ℓ·w/2), and v1's "nearly 6X faster" kernel claim was
// dropped in v2 — so it is treated as retracted and not repeated.
import { el, slider, toggle } from "../lib/dom.js";
import { attentionGrid } from "../views/grid.js";
import { readout } from "../views/bars.js";
import { dot, fmt } from "../model/ops.js";
import { forward, CONFIG } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose } from "./chrome.js";

export function slidingWindowCard(root, m) {
  let w = 4;
  let dilation = 1;
  let dilateSomeHeads = false;
  let global = null;
  let query = null;

  root.appendChild(
    prose({
      problem:
        "The Sparse Transformer's strided pattern was designed for data with periodic structure — its own motivation came from looking at an image model. Text is not that. Most of what a word needs is a few words away, and a handful of tokens matter to the whole document. A pattern built for rows and columns fits neither case.",
      mechanism:
        "Give every token a band of w neighbours, so the cost grows with the window rather than the context. Stack layers and the reach compounds like a convolution. Widen the band by skipping — a dilated window covers d times the distance for the same number of reads — and Longformer applies that to only two heads of eight, keeping the rest dense so local detail survives. For classification it also marks a few tokens as global, readable by everything and reading everything; that part never appears in its autoregressive model, and this card is causal, so the global control here is a demonstration of why.",
    })
  );

  // ------------------------------------------------------------ 1. the window
  const wSlider = slider({
    label: "window w",
    min: 2,
    max: 16,
    step: 2,
    value: 4,
    oninput: (v) => ((w = v), render()),
  });
  const dSlider = slider({
    label: "dilation",
    min: 1,
    max: 4,
    value: 1,
    format: (v) => (v === 1 ? "none" : `every ${v}${v === 2 ? "nd" : v === 3 ? "rd" : "th"}`),
    oninput: (v) => ((dilation = v), render()),
  });
  const grid = attentionGrid({ onPickRow: (i) => ((query = i), render()), label: "sliding window attention" });
  const winRead = readout([
    { key: "reads", label: "keys read per query" },
    { key: "full", label: "full attention would read" },
    { key: "reach", label: "how far back one layer sees" },
    { key: "saved", label: "saved" },
  ]);
  const winNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "a band instead of a triangle" }),
      el("div", { class: "ctrls" }, [wSlider.node, dSlider.node]),
      grid.node,
      winRead.node,
      winNote,
    ])
  );

  // ------------------------------------------------------- 2. depth, on activations
  const depthProbe = el("div", { class: "ctrls" });
  const depthRead = readout([
    { key: "b1", label: "change after block 1" },
    { key: "b2", label: "change after block 2" },
    { key: "why", label: "verdict" },
  ]);
  const depthNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "depth widens the window — measured on the activations" }),
      depthProbe,
      depthRead.node,
      depthNote,
    ])
  );

  // ------------------------------------------------------- 3. dilation's bargain
  const dilToggle = toggle({
    label: "dilate only 2 heads of 4",
    value: false,
    onchange: (v) => ((dilateSomeHeads = v), render()),
  });
  const dilRead = readout([
    { key: "heads", label: "heads reading with gaps" },
    { key: "cost", label: "keys read per query" },
    { key: "distance", label: "distance covered" },
    { key: "gaps", label: "positions skipped inside that span" },
  ]);
  const dilNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the same cost, further back — and what falls through" }),
      el("div", { class: "ctrls" }, [dilToggle]),
      dilRead.node,
      dilNote,
    ])
  );

  // --------------------------------------------------------- 4. the global hub
  const globalPick = el("div", { class: "ctrls" });
  const globRead = readout([
    { key: "cells", label: "extra cells the hub costs" },
    { key: "covers", label: "words it can actually serve" },
    { key: "verdict", label: "useful here?" },
  ]);
  const globNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "a hub everything can see — and why not in this model" }),
      globalPick,
      globRead.node,
      globNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Cost grows with the window rather than the context: at a fixed w, keys read per query stays flat while full attention climbs with the sentence",
        "Depth compounds the reach for free — each block extends how far information can travel, which this card measures on the activations rather than asserting from the mask",
        "Dilation covers several times the distance for exactly the same number of reads, and applying it to only some heads keeps local detail intact in the others",
        "For encoders, a few global tokens give a cheap escape hatch for document-wide information: dropping them and their separate projections cost 8.3 points on WikiHop",
      ],
      givesUp: [
        "Long-range information has to travel through intermediate tokens, and whatever they did not keep is lost",
        "You choose the window, the dilation and the global tokens before seeing the input",
        "Dilation leaves gaps: the span grows but the positions inside it are skipped",
        "The cache still grows with context — the window bounds the compute, not the history",
        "The ablations are modest: staged windows bought 0.03 bits per character and dilation 0.01, on the authors' own short training runs",
      ],
      chooseWhen:
        "Long documents with local structure, and as one layer type in a hybrid — most production long-context stacks interleave windowed layers with a few full ones. Note that the paper's global-token result is an encoder result; in a decoder the hub only covers what precedes it.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "Each word only looks at its neighbours, so the work stops growing out of control as the text gets longer",
        "Stacking layers lets information travel further than one layer alone could reach, at no extra cost",
        "Skipping every second or third word covers much more ground for the same effort",
        "A few designated words can be made visible to everything, which is how a question reaches a whole document",
      ],
      cons: [
        "Anything far away has to be passed along through the words in between, and they may not have kept it",
        "How wide the window is, and which words are special, is decided by a person in advance rather than learned",
        "Skipping words to reach further means the skipped ones are simply not read",
        "It reduces the work but not the notes kept while writing, which still grow with the length of the conversation",
      ],
      verdict:
        "The shape that fits text: mostly local, with a few things everyone needs to see. It makes long documents affordable and leaves you choosing the window by hand — and the choosing, rather than the window, is what the rest of this timeline keeps trying to automate.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    if (query === null || query >= T) query = T - 1;
    wSlider.set(w);
    dSlider.set(dilation);

    const half = Math.floor(w / 2);
    const inWindow = (i, j, d) => {
      const back = i - j;
      return back >= 0 && back % d === 0 && back <= d * half;
    };
    const readable = (i, j, at = {}) => {
      const d = dilateSomeHeads ? (at.head >= CONFIG.HEADS / 2 ? dilation : 1) : dilation;
      if (global !== null && (j === global || i === global) && j <= i) return true;
      return inWindow(i, j, d);
    };

    const res = forward(tokens, { mixer: softmaxMixer({ readable }) });
    const h = res.trace[0].heads[0];
    grid.update({ tokens, weights: h.weights, query, readable: (i, j) => readable(i, j, { head: 0 }) });

    // --- 1. the bill
    let reads = 0;
    for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) if (readable(i, j, { head: 0 })) reads++;
    const full = (T + 1) / 2;
    winRead.update({
      reads: fmt(reads / T, 2),
      full: fmt(full, 2),
      reach: `${dilation * half} words`,
      saved: `${Math.round((1 - reads / T / full) * 100)}%`,
    });
    winNote.innerHTML = `Each query reads about ${fmt(reads / T, 2)} keys where full attention would read ${fmt(full, 2)}. Now lengthen the sentence at the top of the page without touching the window: this number stays pinned near w/2 + 1 while the full-attention figure climbs with the sentence. That is the difference between cost growing with the window and cost growing with the context — measured here, not asserted.`;

    // --- 2. depth, measured on activations
    const far = Math.max(0, query - dilation * half - 1);
    const pick = (label, value, onChange) =>
      el("div", { class: "ctrl" }, [
        el("label", { text: label }),
        el(
          "select",
          { onchange: (e) => onChange(Number(e.target.value)) },
          tokens.map((t, i) => el("option", { value: i, text: `${i}. ${t.word}`, selected: i === value ? "" : null }))
        ),
      ]);
    depthProbe.replaceChildren(pick("watch this word", query, (v) => ((query = v), render())));

    // Change a word outside the window and see how far the effect reaches, block by block.
    const edited = tokens.map((t, i) => (i === far ? { ...tokens[(far + 5) % T] } : t));
    const resB = forward(edited, { mixer: softmaxMixer({ readable }) });
    const norm = (a, b) => {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
      return Math.sqrt(s);
    };
    const d1 = norm(res.trace[0].hidden[query], resB.trace[0].hidden[query]);
    const d2 = norm(res.trace[1].hidden[query], resB.trace[1].hidden[query]);
    depthRead.update({
      b1: d1 < 1e-9 ? "0.000" : fmt(d1, 3),
      b2: d2 < 1e-9 ? "0.000" : fmt(d2, 3),
      why: d1 < 1e-9 && d2 > 1e-9 ? "reached it at layer 2" : d1 > 1e-9 ? "inside the window" : "out of reach",
    });
    depthNote.className = "note " + (d1 < 1e-9 && d2 > 1e-9 ? "good" : "");
    depthNote.textContent =
      d1 < 1e-9 && d2 > 1e-9
        ? `“${tokens[far].word}” sits outside “${tokens[query].word}”'s window, so changing it moves nothing after block 1 — the distance is exactly 0.000. After block 2 it moves by ${fmt(d2, 3)}. The information arrived through a token in between, which is the receptive field growing with depth, demonstrated on the activations rather than drawn on a mask. With ${CONFIG.BLOCKS} blocks and a causal half-window of ${half}, one layer reaches ${dilation * half} words and the stack reaches about ${CONFIG.BLOCKS * dilation * half}. Note the paper's ℓ×w figure is the bidirectional case; causal attention gets half of it.`
        : d1 > 1e-9
        ? `“${tokens[far].word}” is inside the window, so it moves “${tokens[query].word}” immediately — ${fmt(d1, 3)} after the first block. Widen the gap between the two words, or narrow the window, to see the layer-by-layer version.`
        : `Changing “${tokens[far].word}” does not reach “${tokens[query].word}” within this depth at all. With more blocks it eventually would; with ${CONFIG.BLOCKS}, it does not.`;

    // --- 3. dilation
    let gaps = 0;
    for (let back = 1; back <= dilation * half; back++) if (back % dilation !== 0) gaps++;
    const dilatedHeads = dilation === 1 ? 0 : dilateSomeHeads ? CONFIG.HEADS / 2 : CONFIG.HEADS;
    dilRead.update({
      heads: `${dilatedHeads} of ${CONFIG.HEADS}`,
      cost: fmt(reads / T, 2),
      distance: `${dilation === 1 ? half : dilation * half} words${dilateSomeHeads && dilation > 1 ? ` (dense heads still ${half})` : ""}`,
      gaps: String(gaps),
    });
    dilNote.textContent =
      dilation === 1
        ? dilateSomeHeads
          ? "The split is switched on, but dilation is still 1 — so \"every 1st position\" is just a solid window, and the two halves of the heads are reading identically. Raise the dilation above 1 to make the split mean something. Longformer's point is precisely that you do not have to choose one or the other for the whole model."
          : "No dilation: the window is a solid run of neighbours. Raise it and watch the distance grow while the number of keys read stays where it is."
        : `Reading every ${dilation}${dilation === 2 ? "nd" : dilation === 3 ? "rd" : "th"} position covers ${dilation * half} words for the same ${fmt(reads / T, 2)} reads per query — but ${gaps} positions inside that span are skipped entirely. Longformer's answer is not to dilate everything: it dilates two heads of eight and leaves the rest solid, so some heads keep the fine detail while others reach further. The ablation credits dilation with 0.01 bits per character, which is worth knowing before treating it as a major win.`;

    // --- 4. the global hub
    globalPick.replaceChildren(
      el("div", { class: "ctrl" }, [
        el("label", { text: "global token" }),
        el(
          "select",
          { onchange: (e) => ((global = e.target.value === "" ? null : Number(e.target.value)), render()) },
          [
            el("option", { value: "", text: "none", selected: global === null ? "" : null }),
            ...tokens.map((t, i) =>
              el("option", { value: i, text: `${i}. ${t.word}`, selected: i === global ? "" : null })
            ),
          ]
        ),
      ])
    );

    if (global === null) {
      globRead.update({ cells: "0", covers: "—", verdict: "no hub set" });
      globNote.textContent =
        "Pick a word to make global. In an encoder it would become readable by everything and read everything — that is how a question reaches a whole document. Watch what happens to that idea under a causal mask.";
    } else {
      const covers = T - 1 - global;
      globRead.update({
        cells: String(T),
        covers: `${covers} of ${T - 1}`,
        verdict: global === 0 ? "no — it can see nothing" : covers === 0 ? "no — nothing can see it" : "partly",
      });
      globNote.className = "note " + (global === 0 || covers === 0 ? "warn" : "");
      globNote.textContent =
        global === 0
          ? `“${tokens[0].word}” is at position 0, so as a hub it is worthless: everything can read it, but it can read nothing — there is nothing before it. This is the honest limitation of the idea in a decoder, and it is why Longformer's global attention appears only in its bidirectional models. The word "global" does not occur anywhere in the paper's autoregressive section.`
          : `The hub costs a fixed ${T} extra cells wherever you put it and whatever the window is — one column plus one row. But under a causal mask it can only serve the ${covers} words that come after it, and can only gather from the ${global} before it. A hub is a bidirectional idea; a decoder gets half of one.`;
    }
  }

  return { update: render, unmount: () => {} };
}
