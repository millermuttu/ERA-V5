// Two savings that are often confused. Compression changes how many positions are stored;
// top-k selection changes how many of those the expensive read touches.
export function compression({ tokens, blockSize, topBlocks }) {
  const stored = Math.ceil(tokens / blockSize);
  const read = Math.min(topBlocks, stored);
  return { stored, read, tokensRead: read * blockSize };
}

export function mountCompress(el) {
  el.insertAdjacentHTML(
    "beforeend",
    `<div class="demo">
      <div class="demo-title">Store fewer positions, then read fewer of those</div>
      <div class="ctrls">
        <div class="ctrl"><label for="k-m">tokens per block <span id="k-mv">16</span></label>
          <input id="k-m" type="range" min="1" max="64" step="1" value="16"></div>
        <div class="ctrl"><label for="k-k">blocks read per query <span id="k-kv">8</span></label>
          <input id="k-k" type="range" min="1" max="64" step="1" value="8"></div>
      </div>
      <div class="bars" id="k-bars"></div>
      <p class="note" id="k-note"></p>
    </div>`
  );

  const tokens = 32768;
  const $ = (s) => el.querySelector(s);

  function render() {
    const blockSize = Number($("#k-m").value);
    const topBlocks = Number($("#k-k").value);
    $("#k-mv").textContent = blockSize;
    $("#k-kv").textContent = topBlocks;

    const { stored, read, tokensRead } = compression({ tokens, blockSize, topBlocks });
    const bar = (label, val, max, alt) => `
      <div class="bar"><span>${label}</span>
        <span class="track"><span class="fill${alt ? " alt" : ""}" style="width:${Math.min(100, (val / max) * 100).toFixed(2)}%"></span></span>
        <span class="val">${val.toLocaleString()}</span></div>`;

    $("#k-bars").innerHTML =
      bar("positions, uncompressed", tokens, tokens) +
      bar("positions stored", stored, tokens) +
      bar("positions read per query", read, tokens, true) +
      bar("original tokens those stand for", tokensRead, tokens, true);

    $("#k-note").innerHTML =
      `At ${tokens.toLocaleString()} tokens: ${blockSize} per block leaves ${stored.toLocaleString()} summaries stored, ` +
      `and the query runs exact attention over ${read} of them. Those two knobs pay different bills — ` +
      `compression is a memory saving, selection is a compute saving. The catch is ranking: if choosing the best blocks ` +
      `meant attending over all of them, nothing was gained, which is why a cheap low-rank indexer does the ranking and ` +
      `the expensive attention only sees the shortlist. Widen the blocks far enough and one summary is speaking for a ` +
      `paragraph — detail the model can no longer recover.`;
  }

  el.querySelectorAll("input").forEach((c) => (c.oninput = render));
  render();
}
