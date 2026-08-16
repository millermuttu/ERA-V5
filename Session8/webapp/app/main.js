import { el, clear } from "./lib/dom.js";
import { PRESETS } from "./model/vocab.js";
import { state, setText } from "./runner.js";
import { mountTimeline } from "./timeline.js";
import { createDeck } from "./deck.js";
import { CARDS } from "./cards/index.js";

const input = document.getElementById("sentence");
input.value = state.text;
input.addEventListener("input", () => setText(input.value));

const presets = document.getElementById("presets");
clear(presets);
for (const p of PRESETS) {
  presets.appendChild(
    el("button", {
      class: "preset",
      type: "button",
      text: p.split(" ").slice(0, 4).join(" ") + "…",
      title: p,
      onclick: () => {
        input.value = p;
        setText(p);
      },
    })
  );
}

const updateTimeline = mountTimeline(document.getElementById("timeline"), {
  onPick: (i) => deck.show(i),
});

const deck = createDeck({
  stage: document.getElementById("stage"),
  pos: document.getElementById("pos"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  cards: CARDS,
  onChange: (i) => updateTimeline(i),
});

if (new URLSearchParams(location.search).has("selfcheck")) {
  const { runSelfCheck } = await import("./lib/selfcheck.js");
  runSelfCheck();
}
