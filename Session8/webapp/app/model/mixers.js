// The seam. A block hands a mixer the per-token queries, keys and values for one head and gets
// back the mixed output plus whatever the views need to draw. Every mechanism in the timeline is
// a configuration of one of these two, which is why the comparison between them is fair: same
// weights, same sentence, same everything except the rule being explained.
import { dot, softmax } from "./ops.js";

/**
 * Softmax attention.
 *
 *   readable(i, j)   false hides key j from query i entirely (sparse patterns, windows, top-k)
 *   bias(i, j, q, at)  added to the score before softmax. The query vector is passed because
 *                    Shaw et al.'s relative term is q_i · w_clip(j−i), not a scalar per offset;
 *                    ALiBi ignores the query but needs `at.head`, because its slope is per head.
 *                    Note it is added after the scaling, not divided by sqrt(d_k).
 *   rotate(v, pos)   applied to q and k before the dot product (RoPE and its extensions)
 *
 * Defaults are plain causal attention — the baseline the whole app is measured against.
 */
export function softmaxMixer({ readable = null, bias = null, rotate = null } = {}) {
  return function mix(Q, K, V, dh, at = {}) {
    const T = Q.length;
    const scale = Math.sqrt(dh);
    const scores = [];
    const weights = [];
    const out = [];
    let reads = 0;

    for (let i = 0; i < T; i++) {
      const q = rotate ? rotate(Q[i], i) : Q[i];
      const row = new Array(T).fill(-Infinity);
      for (let j = 0; j <= i; j++) {
        if (readable && !readable(i, j, at)) continue;
        const k = rotate ? rotate(K[j], j) : K[j];
        row[j] = dot(q, k) / scale + (bias ? bias(i, j, q, at) : 0);
        reads++;
      }
      const w = softmax(row);
      const o = new Float64Array(dh);
      for (let j = 0; j <= i; j++) {
        if (w[j] === 0) continue;
        for (let d = 0; d < dh; d++) o[d] += w[j] * V[j][d];
      }
      scores.push(row);
      weights.push(w);
      out.push(o);
    }
    return { out, scores, weights, reads, kind: "softmax" };
  };
}

/**
 * Fixed-size recurrent state — the linear-attention family.
 *
 *   write: "add"    S += v kᵀ                accumulate, never correct
 *          "delta"  S += β(v − S k) kᵀ       read first, write only the difference
 *          "gated"  S = gS + β(v − gS k)kᵀ   as delta, but the past decays first
 *
 * `phi` is the feature map standing in for the exponential softmax removed. The state is dh x dh
 * whatever the sequence length, which is the entire point.
 *
 * Two switches the delta paper (Schlag et al. §4.2, App. A.2) requires, and which the add-only
 * rule does not want:
 *
 *   sumNorm   divide φ(k) and φ(q) by the sum of their components before anything uses them
 *             (Eq. 29). It is not cosmetic scaling: A.2 derives it as exactly the condition that
 *             the amount a delta write removes equals the amount it writes. Without it the delta
 *             step is a gradient step with learning rate β‖φ(k)‖², which on this model's real keys
 *             averages 33.9 — far outside the stable band (0, 2) — and the state reaches 3.4e18 by
 *             token 16. The paper's own words: "the models diverged otherwise".
 *   attnNorm  divide the read by z·φ(q), the running denominator carried over from linear
 *             attention. Eq. 25 has no such division, and the paper rejects it for delta models:
 *             "It was crucial to remove the attention normalisation for the Delta Net". Kept on by
 *             default only for the add rule, whose Eq. 19 does have it.
 */
/** The paper's feature map: elu(x) + 1. Non-negative everywhere, and unlike relu it has no
 *  zero-gradient region — which is the stated reason they rejected relu. */
export const elu1 = (x) => (x > 0 ? x + 1 : Math.exp(x));

export function stateMixer({
  write = "add",
  decay = 1,
  beta = 1,
  phi = elu1,
  features = null,
  sumNorm = write !== "add",
  attnNorm = write === "add",
} = {}) {
  // `features` is a vector-valued map R^dh -> R^m, which is what a random-feature method needs:
  // Performer's φ projects into m random directions, so the state is m x dh, not dh x dh. When it
  // is absent the elementwise `phi` applies and m === dh, which is the linear-attention case.
  return function mix(Q, K, V, dh, at = {}) {
    const T = Q.length;
    const base = features || ((v) => Float64Array.from(v, phi));
    // Eq. 29 — divide by the sum of the components, not by the norm. App. A.2 says why: the sum
    // is what balances the write against the remove. A feature map that can go negative can sum
    // to zero, so the divisor is floored rather than trusted.
    const map = sumNorm
      ? (v) => {
          const f = base(v);
          let s = 0;
          for (let i = 0; i < f.length; i++) s += f[i];
          const d = Math.abs(s) < 1e-9 ? (s < 0 ? -1e-9 : 1e-9) : s;
          for (let i = 0; i < f.length; i++) f[i] /= d;
          return f;
        }
      : base;
    const m = map(Q[0]).length;
    let S = Array.from({ length: dh }, () => new Float64Array(m));
    const out = [];
    const snapshots = [];
    const denominators = [];
    let norm = new Float64Array(m);

    for (let i = 0; i < T; i++) {
      const k = map(K[i]);
      const q = map(Q[i]);
      const g = write === "gated" ? decay : 1;

      // read what the state currently returns for this key
      const cur = new Float64Array(dh);
      for (let a = 0; a < dh; a++) cur[a] = dot(S[a], k, 0, m) * g;

      for (let a = 0; a < dh; a++) {
        // Eq. 24: the correction is scaled by the write strength, and nothing else is. β never
        // multiplies S — that would be Peng et al.'s gated rule, which App. B argues against.
        const target = write === "add" ? V[i][a] : beta * (V[i][a] - cur[a]);
        for (let b = 0; b < m; b++) S[a][b] = S[a][b] * g + target * k[b];
      }
      for (let b = 0; b < m; b++) norm[b] = norm[b] * g + k[b];

      // The denominator, kept as computed. Taking its absolute value would quietly rescue a
      // negative normaliser — and a negative normaliser is exactly what a feature map that is not
      // non-negative produces. The requirement is visible only if the failure is left visible.
      const o = new Float64Array(dh);
      let z = 0;
      for (let b = 0; b < m; b++) z += norm[b] * q[b];
      const safe = attnNorm ? (Math.abs(z) < 1e-9 ? (z < 0 ? -1e-9 : 1e-9) : z) : 1;
      for (let a = 0; a < dh; a++) o[a] = dot(S[a], q, 0, m) / safe;
      denominators.push(z);
      out.push(o);

      snapshots.push(S.map((r) => Float64Array.from(r)));
    }
    // A state model reads one object per token, not a growing list of keys.
    return { out, scores: null, weights: null, reads: T, state: S, snapshots, denominators, m, kind: "state" };
  };
}

/**
 * Key/value head sharing. `groups` key/value heads serve `heads` query heads; groups === heads is
 * ordinary multi-head, groups === 1 is multi-query. Returns which kv head a query head reads.
 */
export const kvHeadFor = (head, heads, groups) => Math.floor(head / (heads / groups));
