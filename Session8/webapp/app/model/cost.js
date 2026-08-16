// What a mechanism costs. Two scales, deliberately kept apart:
//   toy      what the model on this page actually did, counted during the forward pass
//   serving  the same formulas at a real deployment size, which is where the numbers get scary
import { CONFIG, DH } from "./transformer.js";

/** Bytes of key/value cache for one conversation. The 2 is one key plus one value. */
export function cacheBytes({ layers, kvHeads, headDim, tokens, batch = 1, bytesPerNumber = 2 }) {
  return 2 * layers * kvHeads * headDim * tokens * batch * bytesPerNumber;
}

// The lesson's worked example, and the check that our formula is the same one.
export const SERVING = {
  layers: 48,
  kvHeads: 8,
  headDim: 128,
  tokens: 32768,
  batch: 1,
  bytesPerNumber: 2,
};

export const GB = 1e9;

/** What the page's own model spent on this input under this mechanism. */
export function toyCost(result, { kvGroups = CONFIG.HEADS, bytesPerNumber = 2, latentDim = null } = {}) {
  const T = result.tokens.length;
  const heads = result.trace[0].heads;
  const stateKind = heads[0].kind === "state";

  const readsPerQuery = heads.reduce((s, h) => s + h.reads, 0) / heads.length / T;
  const fullReads = (T + 1) / 2;

  // A state mixer keeps one dh x dh matrix per head instead of a growing list of vectors.
  const perTokenNumbers = stateKind ? 0 : 2 * kvGroups * DH;
  const fixedNumbers = stateKind ? CONFIG.HEADS * DH * DH : 0;
  const stored = latentDim !== null ? 2 * latentDim * T : perTokenNumbers * T;

  return {
    tokens: T,
    readsPerQuery,
    fullReads,
    cacheNumbers: (stored + fixedNumbers) * CONFIG.BLOCKS,
    cacheBytes: (stored + fixedNumbers) * CONFIG.BLOCKS * bytesPerNumber,
    growsWithContext: !stateKind,
    // Mixing work: pairwise for attention, linear for a fixed state.
    mixOps: stateKind ? T * DH * DH * CONFIG.HEADS : heads.reduce((s, h) => s + h.reads, 0) * DH,
  };
}

/** The same shape of calculation at deployment scale. */
export function servingCost(overrides = {}) {
  const cfg = { ...SERVING, ...overrides };
  const one = cacheBytes({ ...cfg, batch: 1 });
  return {
    cfg,
    perUserGB: one / GB,
    totalGB: (one * cfg.batch) / GB,
  };
}
