// Concept 5 — segment recurrence (Transformer-XL).
// Built from docs/research/transformer-xl.md. The research corrected the headline: the abstract's
// "80% longer than RNNs and 450% longer than vanilla Transformers" fuses two different model
// groups, and the 1,874x figure is the top of a range against a slide-by-one baseline.
import { el, slider, toggle } from "../lib/dom.js";
import { readout } from "../views/bars.js";
import { dot, fmt } from "../model/ops.js";
import { forward, CONFIG } from "../model/transformer.js";
import { softmaxMixer } from "../model/mixers.js";
import { sinusoidalVector } from "../model/position.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

// A weight can be genuinely small; rounding it to 0.000 would read as "not connected", which is
// the exact distinction this card is about.
const tiny = (w) => (w >= 0.001 ? fmt(w, 3) : w > 0 ? w.toExponential(1) : "0.000");

export function transformerXLCard(root, m) {
  let L = 4;
  let M = 0;
  let relative = true;
  let query = null;
  let target = null;

  root.appendChild(
    prose({
      problem:
        "To train on text longer than the window you cut it into segments. With everything so far, each segment starts from nothing: the first token of segment two cannot see the last token of segment one. The paper calls this context fragmentation — the model is handed amnesia at a boundary that has nothing to do with the text, and the first few tokens of every segment are predicted with almost no context.",
      mechanism:
        "Cache the previous segment's hidden states and let the current segment attend over them, with no gradient crossing the boundary. Because the cached states came from different absolute positions, absolute encoding breaks under the reuse — two tokens from different segments would carry the same position vector — so the same paper had to replace it with a relative scheme, keeping the sinusoid rather than learning a table.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root);

  // ---------------------------------------------------------- 1. the cut
  const lSlider = slider({
    label: "segment length",
    min: 2,
    max: 8,
    step: 2,
    value: 4,
    oninput: (v) => ((L = v), render()),
  });
  const mSlider = slider({
    label: "cached memory",
    min: 0,
    max: 3,
    value: 0,
    format: (v) => (v === 0 ? "none" : `${v} segment${v > 1 ? "s" : ""}`),
    oninput: (v) => ((M = v), render()),
  });
  const gridHolder = el("div", { class: "segwrap" });
  const fragRead = readout([
    { key: "starved", label: "words that can see fewer than two others" },
    { key: "reads", label: "keys read, averaged over the sentence" },
    { key: "full", label: "full attention would read" },
  ]);
  const fragNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "cut the sentence into segments" }),
      el("div", { class: "ctrls" }, [lSlider.node, mSlider.node]),
      gridHolder,
      fragRead.node,
      fragNote,
    ])
  );

  // ------------------------------------------------------ 2. reachability probe
  const probe = el("div", { class: "ctrls" });
  const probeRead = readout([
    { key: "reach", label: "reachable?" },
    { key: "weight", label: "attention weight it actually receives" },
    { key: "flip", label: "memory needed to reach it" },
  ]);
  const probeNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "can this word still see that word?" }),
      probe,
      probeRead.node,
      probeNote,
    ])
  );

  // ------------------------------------------------------ 3. frozen vs live
  const sgRead = readout([
    { key: "frozen", label: "keys that came from cache" },
    { key: "live", label: "keys from the current segment" },
    { key: "reach", label: "how far back the stack can reach" },
  ]);
  const sgNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the memory is read-only" }),
      sgRead.node,
      sgNote,
    ])
  );

  // ------------------------------------------ 4. why absolute position breaks
  const relToggle = toggle({
    label: "relative position",
    value: true,
    onchange: (v) => ((relative = v), render()),
  });
  const collideRead = readout([
    { key: "collide", label: "distance between two positions' vectors" },
    { key: "which", label: "which two" },
  ]);
  const collideNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "why the old positional scheme could not survive this" }),
      el("div", { class: "ctrls" }, [relToggle]),
      collideRead.node,
      collideNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "Context reaches past the segment boundary, so the first tokens of a segment are no longer predicted blind",
        "Reuse rather than recomputation at evaluation time — reported up to 1,874× faster per token than sliding the window along by one and recomputing, though that is the top of a range measured against a deliberately slow baseline",
        "The gain is not only about long documents: on shuffled sentences with no long-range structure at all, recurrence alone moved perplexity from 27.1 to 25.2, which isolates fragmentation from long context",
        "Evaluation memory can be longer than training memory — 384 to 1,600 on WikiText-103",
      ],
      givesUp: [
        "The cache is read-only. No gradient crosses the boundary, so nothing ever learns what is worth keeping — it keeps the most recent states because they are the most recent",
        "Cached states cost memory that grows with how far back you keep, and cached tokens are attended to but never attend",
        "Reach is bounded at roughly the number of layers times the segment length, because the recurrence shifts one layer down per segment — not unlimited context",
        "It forced a change of positional scheme, so the two mechanisms arrive entangled",
      ],
      chooseWhen:
        "Streaming or chunked text where the boundary is an artefact of your batching rather than of the document. The idea returns at the end of this timeline as a single gated vector carried between chunks — the same trick compressed as far as it will go.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "The model stops forgetting everything each time it starts a new chunk of text",
        "Reading a long document gets dramatically faster, because earlier work is reused instead of redone",
        "It helps even when the text has no long-range structure, simply because chunk boundaries stop hurting",
      ],
      cons: [
        "The remembered part is frozen: the model can read it but never learns anything from it, so it keeps whatever is most recent rather than whatever is most useful",
        "Remembering more costs more memory, and there is a ceiling on how far back it can reach no matter how much you keep",
        "It only works if the way positions are handled is changed at the same time, so you cannot adopt it on its own",
      ],
      verdict:
        "The first serious attempt to let a model remember something from before the current window, and the compromise it makes is the one everything after it inherits: what you carry forward is cheap, and it is chosen for you by recency rather than learned.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    lSlider.set(L);
    mSlider.set(M);
    if (query === null || query >= T) query = Math.min(T - 1, L + 1);
    if (target === null || target >= T) target = 0;

    const segOf = (i) => Math.floor(i / L);
    const segStart = (i) => segOf(i) * L;
    // Same segment, or inside the M cached positions immediately before this segment starts.
    const readable = (i, j) => j >= segStart(i) - M * L && j <= i;

    const res = forward(tokens, { mixer: softmaxMixer({ readable }) });
    const h = res.trace[0].heads[0];

    flow.update({
      tokens,
      head: { ...h, emb: res.trace[0].input },
      weights: h.weights,
      out: h.out,
      top: res.top,
      query,
      opts: { readable },
    });
    flowNote.innerHTML =
      M === 0
        ? `The small marks are pairs behind a segment wall — the query is not allowed to read them at all. Notice how many of them there are, and that the first word of each segment has almost nothing to its left. Raise the memory below and watch marks come back to life.`
        : `Carrying ${M} segment${M > 1 ? "s" : ""} of memory, pairs that were unreadable a moment ago now carry real weight. Nothing about the projections changed — the same queries, keys and values — only which of them the query is permitted to reach.`;

    // --- the segmented grid, drawn with the walls visible
    const CELL = 17;
    const LEFT = 96;
    const W = LEFT + T * CELL + 30;
    const H = 20 + T * CELL + 20;
    const cells = [];
    for (let i = 0; i < T; i++) {
      for (let j = 0; j <= i; j++) {
        const ok = readable(i, j);
        const fromCache = ok && j < segStart(i);
        const w = h.weights[i][j];
        cells.push(
          `<rect x="${LEFT + j * CELL}" y="${20 + i * CELL}" width="${CELL - 2}" height="${CELL - 2}" rx="2" fill="${
            !ok
              ? "rgba(233,231,220,0.03)"
              : fromCache
              ? `rgba(184,146,62,${(0.15 + Math.min(1, Math.sqrt(w * 3)) * 0.8).toFixed(2)})`
              : `rgba(79,197,140,${(0.1 + Math.min(1, Math.sqrt(w * 3)) * 0.85).toFixed(2)})`
          }"${!ok ? ' stroke="rgba(224,105,61,0.3)"' : ""}/>`
        );
      }
    }
    const walls = [];
    for (let s = L; s < T; s += L) {
      walls.push(
        `<line x1="${LEFT + s * CELL - 1}" y1="14" x2="${LEFT + s * CELL - 1}" y2="${20 + T * CELL}" stroke="#E0693D" stroke-width="1"/>`,
        `<line x1="${LEFT - 4}" y1="${20 + s * CELL - 1}" x2="${LEFT + T * CELL}" y2="${20 + s * CELL - 1}" stroke="#E0693D" stroke-width="1"/>`
      );
    }
    const labels = tokens
      .map(
        (t, i) =>
          `<text x="${LEFT - 8}" y="${20 + i * CELL + 12}" text-anchor="end" class="gridlab${i === query ? " is-current" : ""}">${t.word}</text>`
      )
      .join("");

    gridHolder.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="grid-svg" role="img"
      aria-label="which earlier words each word can read, with segment boundaries and cached memory marked">
      ${cells.join("")}${walls.join("")}${labels}</svg>`;

    // --- fragmentation counts
    let starved = 0;
    let reads = 0;
    for (let i = 0; i < T; i++) {
      let n = 0;
      for (let j = 0; j <= i; j++) if (readable(i, j)) n++;
      reads += n;
      if (n < 2) starved++;
    }
    fragRead.update({
      starved: String(starved),
      reads: fmt(reads / T, 1),
      full: fmt((T + 1) / 2, 1),
    });
    fragNote.className = "note " + (M === 0 && starved > 0 ? "warn" : "");
    fragNote.textContent =
      M === 0
        ? `With no memory, ${starved} of your ${T} words can see fewer than two words — every word that lands at the start of a segment begins with nothing behind it. The red lines are the walls. That is context fragmentation, in your own sentence.`
        : `Carrying ${M} segment${M > 1 ? "s" : ""} of memory, the ochre cells are keys read out of cache rather than recomputed. Words at a segment start now have something behind them, and the average word reads ${fmt(reads / T, 1)} keys against ${fmt((T + 1) / 2, 1)} for full attention.`;

    // --- reachability probe
    const pick = (label, value, onChange) =>
      el("div", { class: "ctrl" }, [
        el("label", { text: label }),
        el(
          "select",
          { onchange: (e) => onChange(Number(e.target.value)) },
          tokens.map((t, i) => el("option", { value: i, text: `${i}. ${t.word}`, selected: i === value ? "" : null }))
        ),
      ]);
    probe.replaceChildren(
      pick("from", query, (v) => ((query = v), render())),
      pick("to", target, (v) => ((target = v), render()))
    );

    const can = target <= query && readable(query, target);
    let flip = null;
    for (let mm = 0; mm <= 6; mm++) {
      if (target >= segStart(query) - mm * L && target <= query) {
        flip = mm;
        break;
      }
    }
    probeRead.update({
      reach: target > query ? "it is in the future" : can ? "yes" : "no",
      weight: can ? tiny(h.weights[query][target]) : "0.000",
      flip: flip === null ? "out of range" : flip === 0 ? "none needed" : `${flip} segment${flip > 1 ? "s" : ""}`,
    });
    probeNote.className = "note " + (can ? "" : "warn");
    probeNote.textContent = can
      ? `“${tokens[query].word}” reads “${tokens[target].word}” with weight ${tiny(h.weights[query][target])} — a real softmax weight from the run above, not a mask entry. Small is not the same as absent: this word is competing for attention, where behind the wall it was not in the running at all.`
      : target > query
      ? `“${tokens[target].word}” comes later in the sentence, so the causal mask hides it regardless of memory.`
      : `“${tokens[query].word}” cannot see “${tokens[target].word}” at all: it sits behind the wall, and no amount of training reaches past a key that was never in the attention. Raise the memory to ${flip} segment${flip > 1 ? "s" : ""} and the weight becomes a number.`;

    // --- frozen vs live
    let frozen = 0;
    let live = 0;
    for (let j = 0; j <= query; j++) {
      if (!readable(query, j)) continue;
      if (j < segStart(query)) frozen++;
      else live++;
    }
    sgRead.update({
      frozen: String(frozen),
      live: String(live),
      reach: `${CONFIG.BLOCKS} × ${M * L} = ${CONFIG.BLOCKS * M * L} back`,
    });
    sgNote.textContent = `Of the ${frozen + live} keys “${tokens[query].word}” reads, ${frozen} came out of cache and ${live} are from its own segment. The cached ones are frozen: the paper's stop-gradient means the loss never travels back through them, so — in its own words — the gradient still remains within a segment. The memory is an activation cache, not something the model learns to curate. And because the recurrence shifts one layer down per segment, the reach is bounded at about the number of layers times the memory, not unlimited.`;

    // --- absolute position collision
    const s1 = sinusoidalVector(0, CONFIG.D);
    const s2 = sinusoidalVector(0, CONFIG.D);
    let dist = 0;
    for (let i = 0; i < s1.length; i++) dist += (s1[i] - s2[i]) ** 2;
    collideRead.update({
      collide: relative ? "not applicable" : fmt(Math.sqrt(dist), 6),
      which: relative
        ? "distance is measured between tokens"
        : `position 0 of segment 1 and position 0 of segment 2`,
    });
    collideNote.className = "note " + (relative ? "" : "warn");
    collideNote.textContent = relative
      ? "With relative position the score depends on how far apart two tokens are, which is well defined across a boundary. Note that Transformer-XL's version keeps the sinusoid and learns only the projections, unlike Shaw's learned table two cards back — and the paper's own ablation shows that difference matters: with Shaw's encoding, giving the model a longer evaluation context bought exactly nothing."
      : `Reuse the cache with absolute positions and the first token of every segment carries the identical position vector — the distance between them is ${fmt(Math.sqrt(dist), 6)}. The model cannot tell a cached token from a current one at the same offset, which is precisely why this mechanism could not be adopted without replacing the positional scheme at the same time.`;
  }

  return { update: render, unmount: () => {} };
}
