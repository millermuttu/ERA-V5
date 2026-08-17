// Concept 7 — multi-query attention.
// Built from docs/research/mqa.md. Two things the research settled: the paper's argument is a
// ratio of memory access to arithmetic, not "the cache is smaller"; and the training-instability
// folklore is not in this paper at all, so it is not claimed here. Concept 15's research located it:
// it is GQA's Appendix A, 2023 — from-scratch multi-query models with frequent loss spikes that
// diverged immediately on long inputs — reported as observed, with no counts, and the authors say
// they stopped looking for the cause. So it belongs to that card, dated 2023, and still not here.
import { el, slider } from "../lib/dom.js";
import { readout, barList } from "../views/bars.js";
import { dot, fmt } from "../model/ops.js";
import { forward, CONFIG, DH } from "../model/transformer.js";
import { cacheBytes, SERVING, GB } from "../model/cost.js";
import { state } from "../runner.js";
import { tradeBlock, plainBlock, prose, flowPanel } from "./chrome.js";

// The paper's incremental-decoding ratios of memory access to arithmetic.
const ratioMHA = (n, d, b) => n / d + 1 / b;
const ratioMQA = (n, d, b, h) => 1 / d + n / (d * h) + 1 / b;

export function mqaCard(root, m) {
  let groups = CONFIG.HEADS;
  let ctx = 15; // log2 tokens for the serving panel
  let batch = 1;

  root.appendChild(
    prose({
      problem:
        "Concept 1 named the second bill and left it: generating one token at a time means keeping every earlier key and value so they need not be recomputed, and with multi-head attention every head stores its own pair. But the paper's argument is sharper than 'that is a lot of memory'. During incremental decoding the accelerator reloads the whole cache to produce a single token, so the ratio of memory access to arithmetic approaches one — and the chip spends its time waiting on memory rather than computing.",
      mechanism:
        "Keep all the query heads, and give the whole layer one shared key head and one shared value head. Every query head still asks its own question; they all ask it of the same stored keys and values. The paper's own summary of the effect on that ratio is that it reduces the offensive n/d term by a factor of h.",
    })
  );

  const { flow, note: flowNote } = flowPanel(root);

  // ------------------------------------------------------------ head sharing
  const groupSlider = slider({
    label: "key/value heads",
    min: 1,
    max: CONFIG.HEADS,
    value: CONFIG.HEADS,
    format: (v) => `${v} for ${CONFIG.HEADS} query heads${v === 1 ? " — multi-query" : v === CONFIG.HEADS ? " — multi-head" : ""}`,
    oninput: (v) => ((groups = v), render()),
  });
  const shareRead = readout([
    { key: "cos", label: "similarity between two heads' keys" },
    { key: "cache", label: "cache numbers per token, this model" },
    { key: "ops", label: "arithmetic per block" },
  ]);
  const shareNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what sharing actually changes" }),
      el("div", { class: "ctrls" }, [groupSlider.node]),
      shareRead.node,
      shareNote,
    ])
  );

  // --------------------------------------------------- the ratio, the real argument
  const ctxSlider = slider({
    label: "context length",
    min: 4,
    max: 20,
    value: 15,
    format: (v) => (2 ** v).toLocaleString(),
    oninput: (v) => ((ctx = v), render()),
  });
  const batchSlider = slider({
    label: "batch size",
    min: 1,
    max: 64,
    value: 1,
    oninput: (v) => ((batch = v), render()),
  });
  const ratioRead = readout([
    { key: "mha", label: "multi-head: memory access ÷ arithmetic" },
    { key: "mqa", label: "multi-query: the same ratio" },
    { key: "gain", label: "improvement" },
  ]);
  const ratioNote = el("p", { class: "note" });

  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "the ratio the paper is actually about" }),
      el("div", { class: "ctrls" }, [ctxSlider.node, batchSlider.node]),
      ratioRead.node,
      ratioNote,
    ])
  );

  // ----------------------------------------------------------- serving cache
  const cacheBars = barList({
    rows: [
      { key: "mha", label: "multi-head, 32 kv heads" },
      { key: "gqa", label: "grouped, 8 kv heads", alt: true },
      { key: "mqa", label: "multi-query, 1 kv head", alt: true },
    ],
  });
  const cacheNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "and what that is worth at serving scale" }),
      cacheBars.node,
      cacheNote,
    ])
  );

  // --------------------------------------------------- what it does NOT show
  const honestRead = readout([
    { key: "rows", label: "how alike the heads' attention rows become" },
    { key: "top", label: "heads agreeing on their top word" },
  ]);
  const honestNote = el("p", { class: "note" });
  root.appendChild(
    el("section", { class: "panel" }, [
      el("div", { class: "panel-title", text: "what this toy cannot show you" }),
      honestRead.node,
      honestNote,
    ])
  );

  root.appendChild(
    tradeBlock({
      buys: [
        "The key/value cache shrinks by the number of heads, which is what turns a memory-bandwidth-bound decode into an arithmetic-bound one",
        "Decoding measured 46 to 3.8 microseconds per token greedy, and 203 to 32 with beam search — a 12× speedup where it matters",
        "Encoder time barely moved, 1.7 to 1.5, and training was a wash at 13.2 to 13.0. That flat encoder is the evidence the gain is about memory bandwidth rather than arithmetic",
        "One integer to change, and it composes with everything else",
      ],
      givesUp: [
        "Quality, measured rather than merely noted: log perplexity 1.424 to 1.439 and dev BLEU 26.7 to 26.5. The paper's own wording is that it is slightly worse than the baseline, but much closer than any of the alternatives involving decreasing the number of heads or their width",
        "All heads must agree on what the past looked like — one stored view of history for every question asked of it",
        "The cache still grows linearly with context; this changes the constant, not the shape",
      ],
      chooseWhen:
        "When decode latency is the constraint and a small, measured quality cost is acceptable — which in 2019 was most serving. GQA in 2023 finds the middle of exactly this trade, and is why you rarely see the fully shared version now.",
    })
  );

  root.appendChild(
    plainBlock({
      pros: [
        "Writing a reply gets about twelve times faster, because the machine spends far less time fetching things from memory",
        "The notes the model keeps while writing shrink by as much as the number of heads it has",
        "It is a one-line change to the design and needs nothing else",
      ],
      cons: [
        "Quality drops slightly, and the paper measured it rather than waving it away",
        "Every one of the model's parallel viewpoints now has to work from a single shared summary of what came before",
        "It makes the memory smaller but does not stop it growing as the conversation gets longer",
      ],
      verdict:
        "A very cheap trade: a large speed-up while writing, for a small, measured loss in quality. Its real lesson is that the bottleneck was never the arithmetic — it was fetching the notes back from memory, and that is why the encoder, which does not have that problem, barely got faster at all.",
    })
  );

  // ------------------------------------------------------------------ render
  function render() {
    const tokens = state.tokens;
    const T = tokens.length;
    if (!T) return;
    groupSlider.set(groups);
    ctxSlider.set(ctx);
    batchSlider.set(batch);

    const res = forward(tokens, { kvGroups: groups });
    const blk = res.trace[0];

    const h0 = blk.heads[0];
    flow.update({
      tokens,
      head: { ...h0, emb: blk.input },
      weights: h0.weights,
      out: h0.out,
      top: res.top,
      opts: { kvShared: groups < CONFIG.HEADS, headFan: { heads: CONFIG.HEADS, groups } },
    });
    flowNote.innerHTML =
      groups === CONFIG.HEADS
        ? `Four heads, each with its own key and value bands. The stacked cards behind the matrix are the other three heads, each running this same picture on its own slice of the width — and each storing its own keys and values while the model writes.`
        : `The key and value bands are now marked shared: ${CONFIG.HEADS} query heads reading ${groups} stored set${groups > 1 ? "s" : ""}. The queries down the left are still four separate projections asking four different questions — nothing about the left of the picture changed. What changed is how many copies of the middle have to be kept in memory while generating.`;

    // how alike are two different query heads' keys now?
    const kA = blk.heads[0].K;
    const kB = blk.heads[CONFIG.HEADS - 1].K;
    let cos = 0;
    for (let i = 0; i < T; i++) {
      const a = kA[i];
      const b = kB[i];
      cos += dot(a, b) / (Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b)) || 1);
    }
    cos /= T;

    const cachePerToken = 2 * groups * DH;
    const ops = blk.heads.reduce((s, h) => s + h.reads, 0) * DH;
    shareRead.update({
      cos: fmt(cos, 4),
      cache: String(cachePerToken),
      ops: ops.toLocaleString(),
    });
    shareNote.innerHTML =
      groups === CONFIG.HEADS
        ? `Every query head has its own key head: two different heads' keys sit at cosine ${fmt(cos, 4)} — unrelated, as separate projections should be. ${cachePerToken} numbers stored per token per block. Slide the control down and watch that number fall while the arithmetic stays exactly where it is.`
        : groups === 1
        ? `All ${CONFIG.HEADS} query heads now read one shared key head, so two heads' keys are cosine <strong>${fmt(cos, 4)}</strong> — the same vectors, by construction. Storage is down to ${cachePerToken} numbers per token, a quarter of multi-head, while the arithmetic is unchanged at ${ops.toLocaleString()} multiplies. Same computation, less memory traffic: that is the whole mechanism.`
        : `${groups} key/value heads serving ${CONFIG.HEADS} query heads — the middle of the trade, which is what GQA formalises four years later. Keys at cosine ${fmt(cos, 4)}, ${cachePerToken} numbers per token.`;

    // --- the ratio
    const n = 2 ** ctx;
    const d = SERVING.headDim * SERVING.kvHeads; // model width at serving scale
    const rMHA = ratioMHA(n, d, batch);
    const rMQA = ratioMQA(n, d, batch, 32);
    ratioRead.update({
      mha: fmt(rMHA, 3),
      mqa: fmt(rMQA, 3),
      gain: `${fmt(rMHA / rMQA, 2)}×`,
    });
    ratioNote.innerHTML = `The paper's incremental-decoding ratios: multi-head is <code>n/d + 1/b</code>, multi-query is <code>1/d + n/(d·h) + 1/b</code>. At ${n.toLocaleString()} tokens and batch ${batch} that is ${fmt(rMHA, 3)} against ${fmt(rMQA, 3)}, an improvement of ${fmt(rMHA / rMQA, 2)}×. The paper's words: when n is close to d, or the batch is close to 1, the ratio is close to 1, <em>causing memory bandwidth to be a major performance bottleneck</em>. Push the batch up and watch the advantage shrink — batching is the other way to fix the same problem, which is why this matters most for single-stream, long-context decoding.`;

    // --- serving cache
    const gb = (kv) => cacheBytes({ ...SERVING, kvHeads: kv, tokens: n, batch }) / GB;
    const worst = gb(32);
    cacheBars.update({
      mha: { value: gb(32), of: worst, text: `${fmt(gb(32), 2)} GB` },
      gqa: { value: gb(8), of: worst, text: `${fmt(gb(8), 2)} GB` },
      mqa: { value: gb(1), of: worst, text: `${fmt(gb(1), 2)} GB` },
    });
    cacheNote.textContent = `48 layers, head dimension 128, bf16, ${n.toLocaleString()} tokens, ${batch} conversation${batch > 1 ? "s" : ""}. Multi-query stores ${fmt(gb(32) / gb(1), 0)}× less than multi-head. Every one of these still doubles when the context doubles — the line gets shallower, it does not stop climbing, which is what GQA and then latent attention are still arguing about years later.`;

    // --- the honest negative result
    let rowSim = 0;
    let agree = 0;
    const q = T - 1;
    for (let a = 0; a < CONFIG.HEADS; a++) {
      for (let b = a + 1; b < CONFIG.HEADS; b++) {
        const wa = blk.heads[a].weights[q];
        const wb = blk.heads[b].weights[q];
        rowSim += dot(wa, wb) / (Math.sqrt(dot(wa, wa)) * Math.sqrt(dot(wb, wb)) || 1);
      }
      const top = blk.heads[a].weights[q].indexOf(Math.max(...blk.heads[a].weights[q]));
      const top0 = blk.heads[0].weights[q].indexOf(Math.max(...blk.heads[0].weights[q]));
      if (top === top0) agree++;
    }
    rowSim /= (CONFIG.HEADS * (CONFIG.HEADS - 1)) / 2;
    honestRead.update({
      rows: fmt(rowSim, 3),
      top: `${Math.round((agree / CONFIG.HEADS) * 100)}%`,
    });
    honestNote.textContent = `It is tempting to expect the heads' attention patterns to collapse into one another once they share keys — and in this model they do not: the similarity between their rows moves only from about 0.33 to 0.37 as you slide all the way to one key head, and the fraction agreeing on a top word barely moves. That is an honest negative result and worth stating. The queries are still separate projections, so the heads still ask different questions; and these weights are untrained, so there is no learned specialisation here to lose. The quality cost the paper measured is real, but it is not visible in a toy this size, and this card will not pretend otherwise.`;
  }

  return { update: render, unmount: () => {} };
}
