// Literal SVG, only where the prose cannot carry the shape. Everything else is text
// plus its trade-off card, which the assignment explicitly allows.

const wrap = (title, svg, note) =>
  `<div class="demo diagram"><div class="demo-title">${title}</div>${svg}<p class="note">${note}</p></div>`;

const CELL = 20;
const grid = (fill) => {
  let out = "";
  for (let i = 0; i < 12; i++) {
    for (let j = 0; j < 12; j++) {
      const c = fill(i, j);
      if (c) out += `<rect x="${40 + j * CELL}" y="${18 + i * CELL}" width="${CELL - 2}" height="${CELL - 2}" fill="${c}" rx="2"/>`;
    }
  }
  return out;
};

const AXES = `
  <text x="4" y="150" transform="rotate(-90 4 150)">query</text>
  <text x="130" y="12">key</text>`;

const JADE = "#4FC58C";
const DIM = "rgba(233,231,220,0.07)";
const OCHRE = "#B8923E";

export const DIAGRAMS = {
  sliding: () =>
    wrap(
      "Who each query is allowed to read",
      `<svg viewBox="0 0 300 270" width="300" height="270" role="img"
            aria-label="attention mask: a diagonal band of allowed positions plus two full columns for global tokens">
        ${grid((i, j) => (j > i ? null : j < 2 ? OCHRE : i - j < 3 ? JADE : DIM))}
        ${AXES}
        <rect x="40" y="262" width="10" height="6" fill="${OCHRE}" rx="1"/><text x="56" y="268">global</text>
        <rect x="110" y="262" width="10" height="6" fill="${JADE}" rx="1"/><text x="126" y="268">window</text>
        <rect x="186" y="262" width="10" height="6" fill="${DIM}" rx="1"/><text x="202" y="268">not read</text>
      </svg>`,
      "Each query reads a band of recent neighbours, so cost grows with the window rather than the context. Two designated columns stay readable to everyone — that is where a question or an instruction lives. Anything in the grey travels only indirectly, one layer at a time, through whatever the tokens in between chose to keep."
    ),

  sinks: () =>
    wrap(
      "Why evicting the oldest tokens breaks the model",
      `<svg viewBox="0 0 300 270" width="300" height="270" role="img"
            aria-label="attention mask for a sliding cache that pins the first few positions as sinks">
        ${grid((i, j) => (j > i ? null : j < 2 ? OCHRE : i - j < 4 ? JADE : DIM))}
        ${AXES}
        <rect x="40" y="262" width="10" height="6" fill="${OCHRE}" rx="1"/><text x="56" y="268">pinned sinks</text>
        <rect x="140" y="262" width="10" height="6" fill="${JADE}" rx="1"/><text x="156" y="268">recent window</text>
      </svg>`,
      "Softmax forces every row to sum to one, so a query with nothing it wants must still put its weight somewhere — and models learn to dump it on the first few positions, which everyone can see. Those tokens carry no meaning; they are a drain. Evict them and the surplus weight redistributes onto real tokens, which is the sudden collapse. Keeping four of them and sliding the rest costs almost nothing and holds quality steady."
    ),

  mla: () =>
    wrap(
      "What actually sits in the cache",
      `<svg viewBox="0 0 560 150" width="560" height="150" role="img"
            aria-label="multi-head, grouped-query and latent caches compared as stored width per token">
        <text x="0" y="26">MHA</text>
        ${Array.from({ length: 8 }, (_, i) => `<rect x="${74 + i * 30}" y="14" width="26" height="16" fill="${JADE}" rx="2"/>`).join("")}
        <text x="330" y="26">8 heads stored</text>

        <text x="0" y="72">GQA</text>
        ${Array.from({ length: 2 }, (_, i) => `<rect x="${74 + i * 30}" y="60" width="26" height="16" fill="${JADE}" rx="2"/>`).join("")}
        ${Array.from({ length: 6 }, (_, i) => `<rect x="${134 + i * 30}" y="60" width="26" height="16" fill="${DIM}" rx="2"/>`).join("")}
        <text x="330" y="72">2 shared, heads agree on the past</text>

        <text x="0" y="118">MLA</text>
        <rect x="74" y="106" width="40" height="16" fill="${OCHRE}" rx="2"/>
        <text x="122" y="118" class="hi">↦ 8 heads rebuilt on read</text>
        <text x="330" y="118">one latent stored</text>
      </svg>`,
      "GQA saves memory by making heads share one stored key and value — the heads lose their individuality to buy the space. MLA stores a compressed latent instead and reconstructs a distinct key and value per head when reading, with the up-projections folded into neighbouring weights so the reconstruction is nearly free. Rotation does not survive that folding, which is why a few dimensions are carried separately just to hold position."
    ),

  nsa: () =>
    wrap(
      "Three branches over the same history",
      `<svg viewBox="0 0 560 168" width="560" height="168" role="img"
            aria-label="compressed blocks, selected blocks and a local window feeding a gate">
        <text x="0" y="14">tokens</text>
        ${Array.from({ length: 21 }, (_, i) => `<rect x="${76 + i * 23}" y="22" width="20" height="12" fill="${DIM}" rx="2"/>`).join("")}

        <text x="0" y="66">compress</text>
        ${Array.from({ length: 6 }, (_, i) => `<rect x="${76 + i * 92}" y="74" width="84" height="12" fill="${JADE}" opacity="0.5" rx="2"/>`).join("")}

        <text x="0" y="110">select</text>
        ${[1, 4].map((b) => `<rect x="${76 + b * 92}" y="118" width="84" height="12" fill="${JADE}" rx="2"/>`).join("")}

        <text x="0" y="154">window</text>
        ${Array.from({ length: 4 }, (_, i) => `<rect x="${440 + i * 23}" y="150" width="20" height="8" fill="${OCHRE}" rx="1"/>`).join("")}
      </svg>`,
      "Runs of tokens are summarised into blocks, so far less is stored. A cheap score over those summaries picks the few blocks worth reading exactly, so far less is attended to. A local window runs alongside, because block granularity is too coarse for the tokens immediately behind you. A learned gate combines the three. Sparsity is present from the first training step, so the model learns to work with it rather than having it imposed at inference."
    ),
};
