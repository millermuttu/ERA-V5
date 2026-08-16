// The parts every concept card carries: the narrative, the technical trade-off record the
// assignment demands, and the plain-language verdict at the foot.
import { el } from "../lib/dom.js";

export function prose({ problem, mechanism }) {
  return el("div", { class: "body" }, [
    el("span", { class: "label", text: "the problem it answered" }),
    el("p", { text: problem }),
    el("span", { class: "label", text: "what it does" }),
    el("p", { text: mechanism }),
  ]);
}

export function tradeBlock({ buys, givesUp, chooseWhen }) {
  const list = (items) => el("ul", {}, items.map((t) => el("li", { text: t })));
  return el("div", { class: "trades" }, [
    el("div", { class: "trade buys" }, [el("span", { class: "label", text: "what it buys" }), list(buys)]),
    el("div", { class: "trade gives" }, [el("span", { class: "label", text: "what it gives up" }), list(givesUp)]),
    el("div", { class: "trade when" }, [
      el("span", { class: "label", text: "when to choose it" }),
      el("p", { text: chooseWhen }),
    ]),
  ]);
}

/**
 * Plain language: no formulas, no symbols, no acronym the sentence does not explain.
 * The self-check compares this with the technical record so a cost cannot quietly go missing.
 */
export function plainBlock({ pros, cons, verdict }) {
  const list = (items, cls) => el("ul", { class: cls }, items.map((t) => el("li", { text: t })));
  return el("div", { class: "plain" }, [
    el("h3", { text: "In plain words" }),
    el("div", { class: "plain-cols" }, [
      el("div", {}, [el("span", { class: "label", text: "what you gain" }), list(pros, "pros")]),
      el("div", {}, [el("span", { class: "label", text: "what it costs you" }), list(cons, "cons")]),
    ]),
    el("p", { class: "verdict", text: verdict }),
  ]);
}
