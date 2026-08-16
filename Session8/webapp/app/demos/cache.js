// The KV-cache bill, one factor at a time. Decimal GB (1e9), which is what the lesson's
// 6.44 GB / 51.54 GB figures are quoted in.
const GB = 1e9;

export function cacheBytes({ layers, kvHeads, headDim, tokens, batch, bytesPerNumber }) {
  return 2 * layers * kvHeads * headDim * tokens * batch * bytesPerNumber;
}

export const LESSON_CONFIG = {
  layers: 48,
  kvHeads: 8,
  headDim: 128,
  tokens: 32768,
  batch: 1,
  bytesPerNumber: 2,
};

const SHARING = { mha: 32, gqa: 8, mqa: 1 };

export function mountCache(el) {
  el.insertAdjacentHTML(
    "beforeend",
    `<div class="demo">
      <div class="demo-title">2 × layers × kv_heads × head_dim × T × batch × bytes</div>
      <div class="ctrls">
        <div class="ctrl"><label for="c-share">head sharing (32 query heads)</label>
          <select id="c-share">
            <option value="mha">MHA — 32 KV heads</option>
            <option value="gqa" selected>GQA — 8 KV heads</option>
            <option value="mqa">MQA — 1 KV head</option>
          </select></div>
        <div class="ctrl"><label for="c-prec">cache precision</label>
          <select id="c-prec"><option value="2" selected>bf16, 2 bytes</option><option value="1">fp8, 1 byte</option></select></div>
        <div class="ctrl"><label for="c-t">context <span id="c-tv"></span></label>
          <input id="c-t" type="range" min="10" max="20" step="1" value="15"></div>
        <div class="ctrl"><label for="c-b">active conversations <span id="c-bv">1</span></label>
          <input id="c-b" type="range" min="1" max="32" step="1" value="1"></div>
      </div>
      <div class="bars" id="c-bars"></div>
      <p class="note" id="c-note"></p>
    </div>`
  );

  const $ = (s) => el.querySelector(s);
  const bars = $("#c-bars");
  const note = $("#c-note");

  function render() {
    const kvHeads = SHARING[$("#c-share").value];
    const bytesPerNumber = Number($("#c-prec").value);
    const tokens = 2 ** Number($("#c-t").value);
    const batch = Number($("#c-b").value);
    $("#c-tv").textContent = tokens.toLocaleString() + " tokens";
    $("#c-bv").textContent = batch;

    const cfg = { ...LESSON_CONFIG, kvHeads, tokens, batch, bytesPerNumber };
    const one = cacheBytes({ ...cfg, batch: 1 }) / GB;
    const all = cacheBytes(cfg) / GB;
    const worst = cacheBytes({ ...cfg, kvHeads: 32, batch: 32, bytesPerNumber: 2, tokens: 2 ** 20 }) / GB;

    const bar = (label, val, alt) => `
      <div class="bar">
        <span>${label}</span>
        <span class="track"><span class="fill${alt ? " alt" : ""}" style="width:${Math.min(100, (val / worst) * 100).toFixed(2)}%"></span></span>
        <span class="val">${val.toFixed(2)} GB</span>
      </div>`;

    bars.innerHTML = bar("one conversation", one) + bar(`× ${batch} conversations`, all, true);
    note.innerHTML = `48 layers, head dim 128. Model weights load once and every user shares them; this does not.
      Double the context and both bars double. Hold the context and double the users and the total doubles too —
      which is why the cache is a per-user serving cost and the weights are not.`;
  }

  el.querySelectorAll("select,input").forEach((c) => (c.oninput = render));
  render();
}
