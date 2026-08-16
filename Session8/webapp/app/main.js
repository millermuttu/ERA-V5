import { byDate, THREADS } from "./data/mechanisms.js";
import { DEMOS } from "./demos/index.js";
import { DIAGRAMS } from "./demos/diagrams.js";

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const showDate = (iso) => {
  const [y, m] = iso.split("-");
  return `${MONTHS[Number(m) - 1]} ${y}`;
};

// What an entry's provenance is called when it is not a verified paper date.
const PROVENANCE = {
  post: "community post, no paper",
  release: "release note, no paper",
  course: "course record only",
  paper: "date not yet verified",
};

function badges(m) {
  const out = [];
  if (m.baseline) out.push(`<span class="badge baseline-flag">the baseline</span>`);
  if (!m.verified) out.push(`<span class="badge">${esc(PROVENANCE[m.source.kind])}</span>`);
  return out.join("");
}

function sourceLine(m) {
  const label = esc(m.source.label);
  return m.source.url
    ? `<p class="src">Source: <a href="${esc(m.source.url)}" rel="noopener">${label}</a></p>`
    : `<p class="src">Source: ${label}</p>`;
}

function card(m, index) {
  const list = (items) => items.map((t) => `<li>${esc(t)}</li>`).join("");
  return `
<article class="card" id="${esc(m.id)}">
  <div class="card-head">
    <span class="date">${showDate(m.date)}</span>
    ${m.baseline ? "" : `<span class="thread" title="${esc(THREADS[m.thread] || "")}">${esc(m.thread)}</span>`}
    ${badges(m)}
  </div>
  <h2>${index}. ${esc(m.name)}</h2>
  ${sourceLine(m)}
  <div class="body">
    <span class="label">The problem it answered</span>
    <p>${esc(m.problem)}</p>
    <span class="label">What it does</span>
    <p>${esc(m.mechanism)}</p>
  </div>
  <div class="trades">
    <div class="trade buys"><span class="label">What it buys</span><ul>${list(m.buys)}</ul></div>
    <div class="trade gives"><span class="label">What it gives up</span><ul>${list(m.givesUp)}</ul></div>
    <div class="trade when"><span class="label">When to choose it</span><p>${esc(m.chooseWhen)}</p></div>
  </div>
  <div class="mount" data-demo="${esc(m.demo || "")}" data-diagram="${esc(m.diagram || "")}"></div>
</article>`;
}

function rail(list) {
  let year = "";
  return list
    .map((m) => {
      const y = m.date.slice(0, 4);
      const head = y === year ? "" : `<div class="year">${(year = y)}</div>`;
      const cls = m.baseline ? ' class="is-baseline"' : "";
      const short = m.name.length > 30 ? m.name.slice(0, 28) + "…" : m.name;
      return `${head}<a href="#${esc(m.id)}"${cls}>${esc(short)}</a>`;
    })
    .join("");
}

const list = byDate();
document.getElementById("rail").innerHTML = rail(list);
document.getElementById("cards").innerHTML = list.map((m, i) => card(m, i + 1)).join("");

// Demos and diagrams mount themselves into the card they belong to.
for (const mount of document.querySelectorAll(".mount")) {
  const { demo, diagram } = mount.dataset;
  if (diagram && DIAGRAMS[diagram]) mount.insertAdjacentHTML("beforeend", DIAGRAMS[diagram]());
  if (demo && DEMOS[demo]) DEMOS[demo](mount);
}

// Cards are built after the document is parsed, so the browser has already tried and failed to
// honour a #fragment by the time they exist. Re-apply it once.
// "instant" because the stylesheet asks for smooth scrolling, and an animated jump here loses
// the race with the browser restoring its own scroll position.
if (location.hash) document.querySelector(location.hash)?.scrollIntoView({ behavior: "instant" });

if (new URLSearchParams(location.search).has("selfcheck")) {
  const { runSelfCheck } = await import("./lib/selfcheck.js");
  runSelfCheck();
}
