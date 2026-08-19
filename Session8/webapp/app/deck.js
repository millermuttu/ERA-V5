// One concept in view. Mount on arrival, unmount on leaving, so a keystroke re-runs one forward
// pass rather than twenty-six.
import { el, clear } from "./lib/dom.js";
import { byDate } from "./data/mechanisms.js";
import { subscribe } from "./runner.js";

const PROVENANCE = {
  post: "community post, no paper",
  release: "release note, no paper",
  course: "course record only",
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const showDate = (iso) => {
  const [y, m] = iso.split("-");
  return `${MONTHS[Number(m) - 1]} ${y}`;
};

export function createDeck({ stage, pos, prev, next, onChange, cards }) {
  const list = byDate();
  let index = 0;
  let teardown = null;
  let unsubscribe = null;

  function header(m, i) {
    return el("div", { class: "card-head" }, [
      el("span", { class: "date", text: showDate(m.date) }),
      i === 0 ? el("span", { class: "badge baseline-flag", text: "the baseline" }) : null,
      m.verified ? null : el("span", { class: "badge", text: PROVENANCE[m.source.kind] }),
      el("span", { class: "seq", text: `${i + 1} of ${list.length}` }),
    ]);
  }

  function show(i, { push = true } = {}) {
    // Clamp rather than wrap: the arrows are the same navigation as the ‹ › buttons, and those are
    // disabled at the ends. Wrapping here made ← on the first card jump to the last one while the
    // button next to it was greyed out.
    index = Math.max(0, Math.min(list.length - 1, i));
    const m = list[index];

    if (teardown) teardown();
    if (unsubscribe) unsubscribe();
    teardown = unsubscribe = null;

    clear(stage);
    const root = el("article", { class: "card", id: m.id });
    root.appendChild(header(m, index));
    root.appendChild(el("h2", { text: m.name }));
    root.appendChild(
      el("p", { class: "src" }, [
        "Source: ",
        m.source.url
          ? el("a", { href: m.source.url, rel: "noopener", text: m.source.label })
          : document.createTextNode(m.source.label),
      ])
    );
    stage.appendChild(root);

    const build = cards[m.id];
    if (build) {
      const mounted = build(root, m) || {};
      teardown = mounted.unmount || null;
      if (mounted.update) {
        unsubscribe = subscribe(mounted.update);
        mounted.update(); // draw once on arrival; subscribe only fires on later changes
      }
    } else {
      root.appendChild(
        el("div", { class: "pendingcard" }, [
          el("p", { text: "This concept has not been built yet." }),
          el("p", {
            class: "note",
            text: "Each card is researched from its primary source and then implemented as a configuration of the model above, one at a time.",
          }),
        ])
      );
    }

    pos.textContent = `${index + 1} / ${list.length} · ${m.name}`;
    prev.disabled = index === 0;
    next.disabled = index === list.length - 1;

    if (push) {
      const url = `#${m.id}`;
      if (location.hash !== url) history.pushState({ id: m.id }, "", url);
    }
    onChange(index, m);
  }

  prev.onclick = () => show(index - 1);
  next.onclick = () => show(index + 1);

  addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    if (e.key === "ArrowLeft") show(index - 1);
    if (e.key === "ArrowRight") show(index + 1);
  });

  addEventListener("popstate", () => {
    const id = location.hash.slice(1);
    const at = list.findIndex((m) => m.id === id);
    show(at < 0 ? 0 : at, { push: false });
  });

  const start = list.findIndex((m) => m.id === location.hash.slice(1));
  show(start < 0 ? 0 : start, { push: false });

  return { show, current: () => list[index] };
}
