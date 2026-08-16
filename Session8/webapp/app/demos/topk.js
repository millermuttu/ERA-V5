// Twelve candidate keys, one query. Keeping the best k cuts what the output sums over —
// and leaves the cost of finding out which k they were exactly where it was.
import { softmax, fmt } from "../lib/mathx.js";

const SCORES = [0.4, 2.1, 0.2, 1.7, 0.1, 0.9, 2.6, 0.3, 1.2, 0.5, 0.7, 1.9];

export function topk(k) {
  const order = SCORES.map((s, i) => i).sort((a, b) => SCORES[b] - SCORES[a]);
  const kept = new Set(order.slice(0, k));
  const weights = softmax(SCORES.map((s, i) => (kept.has(i) ? s : -Infinity)));
  return { kept, weights, valueWork: k, selectionCost: SCORES.length };
}

export function mountTopk(el) {
  el.insertAdjacentHTML(
    "beforeend",
    `<div class="demo">
      <div class="demo-title">Keep the best k, and notice what that cost</div>
      <div class="ctrls">
        <div class="ctrl"><label for="t-k">k = <span id="t-kv">4</span> of 12 keys</label>
          <input id="t-k" type="range" min="1" max="12" step="1" value="4"></div>
      </div>
      <div class="bars" id="t-bars"></div>
      <div class="bars" id="t-cost" style="margin-top:18px"></div>
      <p class="note" id="t-note"></p>
    </div>`
  );

  const slider = el.querySelector("#t-k");

  function render() {
    const k = Number(slider.value);
    el.querySelector("#t-kv").textContent = k;
    const { kept, weights, valueWork, selectionCost } = topk(k);

    el.querySelector("#t-bars").innerHTML = SCORES.map((s, i) => `
      <div class="bar">
        <span>key ${String(i).padStart(2, "0")} · score ${fmt(s, 1)}</span>
        <span class="track"><span class="fill" style="width:${(weights[i] * 100).toFixed(2)}%"></span></span>
        <span class="val">${kept.has(i) ? fmt(weights[i], 3) : "dropped"}</span>
      </div>`).join("");

    el.querySelector("#t-cost").innerHTML = `
      <div class="bar"><span>values summed</span>
        <span class="track"><span class="fill" style="width:${(valueWork / 12) * 100}%"></span></span>
        <span class="val">${valueWork} / 12</span></div>
      <div class="bar"><span>keys scored to choose</span>
        <span class="track"><span class="fill alt" style="width:100%"></span></span>
        <span class="val">${selectionCost} / 12</span></div>`;

    el.querySelector("#t-note").textContent =
      `Drag k down and the top bar shrinks: the expensive weighted sum touches fewer values. The bottom bar never moves. ` +
      `Naive top-k still scores every candidate before it can rank them, so if scoring was the expensive part, nothing has been saved. ` +
      `That gap is why practical sparse attention needs a cheaper proposal step — a window, a router, or a compressed index.`;
  }

  slider.oninput = render;
  render();
}
