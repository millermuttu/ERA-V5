// Two claims from the lesson, computed rather than asserted:
//   1. with softmax off, visiting every old key and reading one pre-built state agree exactly
//   2. an add-only state cannot correct an association; the delta rule can
import { softmax, fmt } from "../lib/mathx.js";

const q = 2;
const keys = [0.5, 1.0, 1.5];
const vals = [10, 20, 30];

// Visit every stored key-value pair, weight, sum.
export function direct(useSoftmax) {
  const raw = keys.map((k) => q * k);
  const w = useSoftmax ? softmax(raw) : raw;
  return w.reduce((s, wi, i) => s + wi * vals[i], 0);
}

// Fold the past into one state before the query arrives, then read it once.
export function regrouped() {
  const S = keys.reduce((s, k, i) => s + k * vals[i], 0);
  return { S, out: q * S };
}

// Add-only write versus the delta rule, on the lesson's numbers.
export const addOnly = (current, wanted) => current + wanted;
export const deltaRule = (current, wanted) => current + (wanted - current);

export function mountLinear(el) {
  el.insertAdjacentHTML(
    "beforeend",
    `<div class="demo">
      <div class="demo-title">Regrouping the sum, and correcting what it stored</div>
      <div class="ctrls">
        <button class="tg" id="l-sm" aria-pressed="false">softmax: off</button>
      </div>
      <div id="l-out"></div>
      <p class="note" id="l-note"></p>
      <hr style="border:0;border-top:1px solid var(--line);margin:22px 0">
      <div class="demo-title">The same state, asked to change its mind</div>
      <div id="l-delta"></div>
    </div>`
  );

  const btn = el.querySelector("#l-sm");
  const out = el.querySelector("#l-out");
  const note = el.querySelector("#l-note");
  let sm = false;

  function render() {
    const d = direct(sm);
    const { S, out: r } = regrouped();
    const agree = Math.abs(d - r) < 1e-9;
    out.innerHTML = `
      <table class="num">
        <tr><td class="rowhead">direct: visit every old key</td><td>${fmt(d, 2)}</td></tr>
        <tr><td class="rowhead">regrouped: read one state S = ${fmt(S, 1)}</td><td>${fmt(r, 2)}</td></tr>
      </table>`;
    note.className = "note " + (agree ? "good" : "warn");
    note.textContent = agree
      ? "Both routes return the same number. With no shared denominator tying the scores together, the query factors out and the past can be summarised before it arrives — that is what makes a fixed-size state possible."
      : "The routes disagree. Softmax divides every score by a sum over all the others, so the weight of one key depends on the rest — nothing can be folded up until the query is known.";
  }

  btn.onclick = () => {
    sm = !sm;
    btn.setAttribute("aria-pressed", String(sm));
    btn.textContent = `softmax: ${sm ? "on" : "off"}`;
    render();
  };
  render();

  const current = 40;
  const wanted = 55;
  el.querySelector("#l-delta").innerHTML = `
    <table class="num">
      <tr><td class="rowhead">key A currently returns</td><td>${current}</td></tr>
      <tr><td class="rowhead">key A should now return</td><td>${wanted}</td></tr>
      <tr><td class="rowhead">add the whole new answer</td><td class="masked">${addOnly(current, wanted)}</td></tr>
      <tr><td class="rowhead">write only the difference (${wanted} − ${current} = ${wanted - current})</td><td class="hot">${deltaRule(current, wanted)}</td></tr>
    </table>
    <p class="note">There is only ever one state matrix. The add-only rule carries the old contribution forward inside it, because the new state was computed from the old one and nothing cancelled what is no longer wanted.</p>`;
}
