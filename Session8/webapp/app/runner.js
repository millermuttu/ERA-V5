// One sentence, one model, one place. The mounted concept subscribes; nothing else computes.
import { tokenize, PRESETS } from "./model/vocab.js";
import { forward } from "./model/transformer.js";

const listeners = new Set();

export const state = {
  text: PRESETS[0],
  tokens: tokenize(PRESETS[0]),
  playing: false,
  block: 0,
  head: 0,
};

/** The baseline result for the current sentence, cached — every card compares against it. */
let baselineCache = null;
export function baseline() {
  if (!baselineCache) baselineCache = forward(state.tokens);
  return baselineCache;
}

/** Run the model under a mechanism configuration. */
export const run = (mech) => forward(state.tokens, mech);

export function setText(text) {
  state.text = text;
  state.tokens = tokenize(text);
  baselineCache = null;
  emit();
}

export function setView({ block, head }) {
  if (block !== undefined) state.block = block;
  if (head !== undefined) state.head = head;
  emit();
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function emit() {
  for (const fn of listeners) fn(state);
}
