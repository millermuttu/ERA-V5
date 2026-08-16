// A real, if tiny, decoder-only transformer. 32 dimensions, 4 heads, 2 blocks, pre-norm,
// tied output embedding, seeded weights. Untrained: the structure is genuine, the knowledge is not.
import { mulberry32, randMat, matvec, add, layerNorm, gelu, softmax } from "./ops.js";
import { V, word } from "./vocab.js";
import { softmaxMixer, kvHeadFor } from "./mixers.js";

export const CONFIG = { D: 32, HEADS: 4, BLOCKS: 2, FF: 64, SEED: 20260817 };
export const DH = CONFIG.D / CONFIG.HEADS;

const rnd = mulberry32(CONFIG.SEED);
const W = {
  emb: randMat(rnd, V, CONFIG.D, 0.5),
  blocks: Array.from({ length: CONFIG.BLOCKS }, () => ({
    q: randMat(rnd, CONFIG.D, CONFIG.D, 0.35),
    k: randMat(rnd, CONFIG.D, CONFIG.D, 0.35),
    v: randMat(rnd, CONFIG.D, CONFIG.D, 0.35),
    o: randMat(rnd, CONFIG.D, CONFIG.D, 0.35),
    f1: randMat(rnd, CONFIG.D, CONFIG.FF, 0.3),
    f2: randMat(rnd, CONFIG.FF, CONFIG.D, 0.3),
  })),
};

const slice = (v, h) => v.subarray(h * DH, (h + 1) * DH);

/**
 * Run the model.
 *
 * `mech` describes the mechanism under test:
 *   mixer     a mixer from mixers.js, defaulting to plain causal softmax attention
 *   position  { add(vec, pos) } applied to the embedding, and/or { rotate } handed to the mixer
 *   kvGroups  key/value head sharing, defaults to one per query head
 *
 * Returns everything the views need — no view recomputes anything.
 */
export function forward(tokens, mech = {}) {
  const T = tokens.length;
  const mixer = mech.mixer || softmaxMixer({});
  const groups = mech.kvGroups || CONFIG.HEADS;

  let h = tokens.map((t, i) => {
    const e = Float64Array.from(W.emb[t.id]);
    return mech.position?.add ? mech.position.add(e, i) : e;
  });

  const trace = [];
  for (let b = 0; b < CONFIG.BLOCKS; b++) {
    const wb = W.blocks[b];
    const normed = h.map(layerNorm);
    const Qf = normed.map((x) => matvec(x, wb.q));
    const Kf = normed.map((x) => matvec(x, wb.k));
    const Vf = normed.map((x) => matvec(x, wb.v));

    const heads = [];
    const mixed = Array.from({ length: T }, () => new Float64Array(CONFIG.D));
    for (let head = 0; head < CONFIG.HEADS; head++) {
      const kvh = kvHeadFor(head, CONFIG.HEADS, groups);
      const Q = Qf.map((x) => slice(x, head));
      const K = Kf.map((x) => slice(x, kvh));
      const Vh = Vf.map((x) => slice(x, kvh));
      const r = mixer(Q, K, Vh, DH);
      for (let i = 0; i < T; i++) mixed[i].set(r.out[i], head * DH);
      heads.push({ ...r, head, kvHead: kvh, Q, K, V: Vh });
    }

    h = h.map((x, i) => add(x, matvec(mixed[i], wb.o)));
    h = h.map((x) => {
      const n = layerNorm(x);
      const f = matvec(n, wb.f1);
      for (let i = 0; i < f.length; i++) f[i] = gelu(f[i]);
      return add(x, matvec(f, wb.f2));
    });

    trace.push({ block: b, heads, hidden: h.map((x) => Float64Array.from(x)) });
  }

  const last = layerNorm(h[T - 1]);
  const logits = W.emb.map((row) => {
    let s = 0;
    for (let i = 0; i < CONFIG.D; i++) s += last[i] * row[i];
    return s;
  });
  const probs = softmax(logits, mech.temperature || 1);
  const top = probs
    .map((p, id) => ({ id, p, word: word(id) }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 8);

  return { tokens, trace, hidden: h, logits, probs, top };
}

/** Sample the next token from the model's own distribution — used by playback. */
export function sample(probs, rand = Math.random) {
  let r = rand();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}
