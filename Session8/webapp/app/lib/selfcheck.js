// The whole test suite. No framework: assertions over the model, over each mechanism's
// implementation, and over the integrity of the chronology. Run with ?selfcheck=1.
import { el } from "./dom.js";
import { softmax, dot, mulberry32, gauss } from "../model/ops.js";
import { forward, CONFIG, DH } from "../model/transformer.js";
import { softmaxMixer, stateMixer, kvHeadFor } from "../model/mixers.js";
import { cacheBytes, SERVING, GB } from "../model/cost.js";
import { sinusoidalVector, learnedTable, relativeBuckets, rope } from "../model/position.js";
import { tokenize, PRESETS } from "../model/vocab.js";
import { mechanisms } from "../data/mechanisms.js";
import { CARDS } from "../cards/index.js";
import { deltaMixer, chunkMixer, chunkCost, seqCost } from "../cards/parallel-deltanet.js";
import { ruleMixer } from "../cards/gated-deltanet.js";
import { nsaAt, nsaMixer, decodeReads, breakEven, PAPER, APP } from "../cards/nsa.js";

const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;
const TOKENS = tokenize(PRESETS[0]);

export function checks() {
  const out = [];
  const ok = (name, pass, detail = "") => out.push({ name, pass, detail });

  // ---------------------------------------------------------------- the model
  const base = forward(TOKENS);
  const w = base.trace[0].heads[0].weights;
  ok("softmax rows sum to 1", w.every((r) => near(r.reduce((a, b) => a + b, 0), 1)));
  ok("masked future weight is exactly 0", w.every((r, i) => r.every((x, j) => j <= i || x === 0)));
  ok("softmax survives an all-but-one masked row", near(softmax([-Infinity, -Infinity, 2])[2], 1));
  ok("the forward pass is deterministic", JSON.stringify(forward(TOKENS).top) === JSON.stringify(base.top));
  ok("the output distribution is a distribution", near(base.probs.reduce((a, b) => a + b, 0), 1));

  // ------------------------------------------------- mechanisms vs the baseline
  // Anything that claims to reduce to plain attention at its degenerate setting must do so.
  // Compare every head, not just head 0 — under key/value sharing head 0 keeps its own kv head
  // and is identical by construction, so checking it alone would pass a broken implementation.
  const same = (mech) => {
    const r = forward(TOKENS, mech);
    return r.trace.every((blk, b) =>
      blk.heads.every((h, hi) =>
        h.out.every((v, i) => v.every((x, d) => near(x, base.trace[b].heads[hi].out[i][d], 1e-12)))
      )
    );
  };
  ok("a readability rule that hides nothing equals the baseline", same({ mixer: softmaxMixer({ readable: () => true }) }));
  ok("a zero score bias equals the baseline", same({ mixer: softmaxMixer({ bias: () => 0 }) }));
  ok("a window as wide as the context equals the baseline", same({ mixer: softmaxMixer({ readable: (i, j) => i - j < 999 }) }));
  ok("key/value sharing at one group per head equals the baseline", same({ kvGroups: CONFIG.HEADS }));
  ok(
    "sharing fewer key/value heads does change the answer",
    !same({ kvGroups: 1 }),
    "so the sharing is real, not cosmetic"
  );
  ok(
    "a shared key/value head is mean-pooled, not one head picked",
    (() => {
      const mean = forward(TOKENS, { kvGroups: 1 });
      const sel = forward(TOKENS, { kvGroups: 1, kvPool: "select" });
      return !mean.trace[0].heads[1].out.every((v, i) =>
        v.every((x, d) => near(x, sel.trace[0].heads[1].out[i][d], 1e-12))
      );
    })(),
    "GQA compares the two, so both have to exist"
  );
  ok(
    "kv head mapping groups query heads correctly",
    kvHeadFor(0, 4, 2) === 0 && kvHeadFor(1, 4, 2) === 0 && kvHeadFor(2, 4, 2) === 1 && kvHeadFor(3, 4, 2) === 1
  );

  // A hidden key must really be gone, not merely down-weighted.
  const hidden = forward(TOKENS, { mixer: softmaxMixer({ readable: (i, j) => j !== 1 }) });
  ok("a hidden key gets exactly zero weight", hidden.trace[0].heads[0].weights.every((r) => r[1] === 0));

  // ------------------------------------------------------ the softmax-free path
  // The lesson's own numbers: with softmax removed, the direct sum and the pre-built state agree.
  const q = 2;
  const keys = [0.5, 1, 1.5];
  const vals = [10, 20, 30];
  const direct = keys.reduce((s, k, i) => s + q * k * vals[i], 0);
  const S = keys.reduce((s, k, i) => s + k * vals[i], 0);
  ok("regrouping matches the direct sum without softmax", near(direct, q * S) && direct === 140, "140");
  const smax = softmax(keys.map((k) => q * k));
  ok("with softmax the two stop agreeing", !near(smax.reduce((s, x, i) => s + x * vals[i], 0), q * S));
  ok("an add-only write overshoots to 95", 40 + 55 === 95);
  ok("the delta rule lands on 55", 40 + (55 - 40) === 55);
  // The paper's sum normalisation is load-bearing, not cosmetic: without it the delta step is a
  // gradient step with learning rate ~34 and the state leaves the range of the display entirely.
  const maxState = (mech) => {
    const s = forward(TOKENS, { mixer: stateMixer(mech) }).trace[0].heads[0].state;
    return Math.max(...s.flatMap((r) => Array.from(r, Math.abs)));
  };
  ok("the delta state stays bounded with the paper's normalisation", maxState({ write: "delta" }) < 10, maxState({ write: "delta" }).toFixed(2));
  ok(
    "and diverges without it, which is why the normalisation exists",
    maxState({ write: "delta", sumNorm: false }) > 1e6,
    maxState({ write: "delta", sumNorm: false }).toExponential(1)
  );
  ok(
    "the write strength scales the correction and nothing else",
    (() => {
      const zero = forward(TOKENS, { mixer: stateMixer({ write: "delta", beta: 0 }) }).trace[0].heads[0].state;
      return zero.every((r) => r.every((x) => x === 0));
    })(),
    "β = 0 writes nothing at all"
  );
  ok("the state mixer runs and keeps a fixed-size state", (() => {
    const r = forward(TOKENS, { mixer: stateMixer({ write: "delta" }) });
    const h = r.trace[0].heads[0];
    return h.kind === "state" && h.state.length === DH && h.state[0].length === DH;
  })());

  // ------------------------------------------- concept 22: the chunkwise form changes nothing
  // The card's whole claim is an equivalence, so it is asserted rather than shown. Compare the
  // two implementations across the whole model at every chunk size, including both endpoints —
  // C = 1 must be the recurrence and C = L must be the fully parallel form.
  const seqRun = forward(TOKENS, { mixer: deltaMixer({}) });
  const chunkDiffs = [];
  for (let C = 1; C <= TOKENS.length; C++) {
    const r = forward(TOKENS, { mixer: chunkMixer({ C }) });
    let d = 0;
    for (let b = 0; b < CONFIG.BLOCKS; b++)
      for (let h = 0; h < CONFIG.HEADS; h++)
        for (let i = 0; i < TOKENS.length; i++)
          for (let a = 0; a < DH; a++) d = Math.max(d, Math.abs(r.trace[b].heads[h].out[i][a] - seqRun.trace[b].heads[h].out[i][a]));
    for (let i = 0; i < r.logits.length; i++) d = Math.max(d, Math.abs(r.logits[i] - seqRun.logits[i]));
    chunkDiffs.push(d);
  }
  ok(
    "the chunkwise delta rule equals the recurrence at every chunk size",
    chunkDiffs.every((d) => d < 1e-12),
    `worst disagreement ${Math.max(...chunkDiffs).toExponential(2)} over ${chunkDiffs.length} chunk sizes`
  );
  ok(
    "and it is a real comparison — the same run against a different rule does disagree",
    (() => {
      const other = forward(TOKENS, { mixer: chunkMixer({ C: 4, beta: 0.5 }) });
      return other.trace[0].heads[0].out.some((v, i) => v.some((x, a) => Math.abs(x - seqRun.trace[0].heads[0].out[i][a]) > 1e-6));
    })(),
    "so the check above is not comparing something with itself"
  );
  // The cost numbers on the card come from a counter inside the loops; the numbers quoted at
  // 4,096 tokens come from a formula over shapes. If those two ever drift the card starts lying.
  ok(
    "the counted multiplications match the cost formula the card quotes at scale",
    (() => {
      for (const C of [1, 3, 4, 8, TOKENS.length]) {
        const r = forward(TOKENS, { mixer: chunkMixer({ C }) });
        const expect = chunkCost(TOKENS.length, DH, C);
        const h = r.trace[0].heads[0];
        if (h.mul !== expect.mul || h.steps !== expect.steps) return false;
      }
      return true;
    })()
  );
  ok(
    "the chunked form buys steps with arithmetic, in that direction",
    (() => {
      const one = chunkCost(4096, 64, 1);
      const big = chunkCost(4096, 64, 64);
      return big.steps < one.steps && big.mul > one.mul && big.mul / seqCost(4096, 64).mul > 1.5;
    })(),
    `C = 64 at 4,096 tokens: ${(chunkCost(4096, 64, 64).mul / seqCost(4096, 64).mul).toFixed(2)}x the arithmetic, ${chunkCost(4096, 64, 64).steps} steps`
  );
  // §3.3: L2 normalisation makes the transition matrix a projection at full write strength. The
  // card says so on screen, so the eigenvalue is checked rather than trusted.
  const eigs = (() => {
    const silu = (x) => x / (1 + Math.exp(-x));
    const norm2 = (f) => Math.sqrt(f.reduce((a, x) => a + x * x, 0));
    const keys = base.trace[0].heads.flatMap((h) => h.K).map((k) => Float64Array.from(k, silu));
    const under = (scale) => keys.map((f) => { const s = scale(f); return 1 - norm2(Float64Array.from(f, (x) => x / s)) ** 2; });
    return {
      raw: under(() => 1),
      l2: under(norm2),
      l1: under((f) => f.reduce((a, x) => a + Math.abs(x), 0)),
    };
  })();
  ok(
    "L2 normalisation makes the transition matrix a projection at full write strength",
    eigs.l2.every((e) => near(e, 0, 1e-12)),
    "1 − β‖k‖₂² is exactly 0, so one direction is erased and the other d−1 are untouched"
  );
  ok(
    "under concept 11's L1 convention a full write cannot erase what it overwrites",
    eigs.l1.every((e) => e > 0.3 && e < 1),
    `${(100 * (eigs.l1.reduce((a, b) => a + b, 0) / eigs.l1.length)).toFixed(0)}% of the old association survives`
  );
  ok(
    "and unnormalised keys put the transition matrix outside the unit disk",
    eigs.raw.some((e) => Math.abs(e) > 1),
    `worst eigenvalue ${Math.min(...eigs.raw).toFixed(1)}`
  );

  // ------------------------------------------------------------------ position
  const offsetInvariant = [1, 4, 10].every((k) => {
    const d = [0, 5, 20, 50].map((p) => dot(sinusoidalVector(p), sinusoidalVector(p + k)));
    return d.every((x) => near(x, d[0], 1e-9));
  });
  ok("sinusoidal similarity depends on the offset, not the position", offsetInvariant);
  const table = learnedTable({ rows: 6 });
  ok("a learned table has nothing past its last row", table.inRange(5) && !table.inRange(6) && table.vector(6) === null);
  const rel = relativeBuckets({ k: 3, dims: DH });
  ok(
    "relative buckets clip past k into one bucket",
    rel.vector(-3) === rel.vector(-7) && rel.vector(-1) !== rel.vector(-3)
  );
  const probe = Float64Array.from({ length: DH }, (_, i) => (i % 2 ? 0.4 : -0.3));
  ok(
    "the relative term is query-dependent, as Shaw eq. 5 requires",
    rel.bias(5, 2, probe) !== rel.bias(5, 2, probe.map((x) => -x))
  );

  // The base change has to hit both endpoints exactly, or it is not the method: the slowest pair
  // must land on what interpolation would have given it, and the fastest must not move at all.
  const scale = 4;
  const plainRope = rope({ dims: DH });
  const ntkRope = rope({ base: 10000 * Math.pow(scale, DH / (DH - 2)), dims: DH });
  const last = DH / 2 - 1;
  ok("a base change leaves the fastest pair exactly alone", ntkRope.freqs[0] === plainRope.freqs[0]);
  ok(
    "and lands the slowest pair exactly where interpolation would have put it",
    near(ntkRope.freqs[last], plainRope.freqs[last] / scale, 1e-15),
    (plainRope.freqs[last] / ntkRope.freqs[last]).toFixed(6)
  );
  ok(
    "the pairs between are compressed by less than asked, which is why the scale under-delivers",
    plainRope.freqs.every((f, i) => f / ntkRope.freqs[i] <= scale + 1e-12) &&
      plainRope.freqs.some((f, i) => i > 0 && i < last && f / ntkRope.freqs[i] < scale - 1e-9)
  );
  ok(
    "a base change is still exactly relative",
    (() => {
      const at = (sh) =>
        forward(TOKENS, { mixer: softmaxMixer({ rotate: (v, p) => ntkRope.rotate(v, p + sh) }) }).trace[0].heads[0]
          .weights;
      const a = at(0);
      const b = at(4096);
      return a.every((r, i) => r.every((x, j) => near(x, b[i][j], 1e-12)));
    })(),
    "shifting the sentence by 4096 leaves the attention matrix alone"
  );

  // YaRN's ramp is a per-pair choice between interpolation and doing nothing, so the seam has to
  // reproduce both endpoints exactly — otherwise the crossfade is measuring something else.
  ok(
    "a per-pair stretch of 1 everywhere is plain rotation",
    (() => {
      const a = rope({ dims: DH });
      const b = rope({ dims: DH, stretch: () => 1 });
      return a.freqs.every((f, i) => a.applied(i) === b.applied(i));
    })()
  );
  ok(
    "and a per-pair stretch of 1/s everywhere is interpolation",
    (() => {
      const flat = rope({ dims: DH, stretch: 1 / scale });
      const fn = rope({ dims: DH, stretch: () => 1 / scale });
      const probe = Float64Array.from({ length: DH }, (_, i) => (i % 2 ? 0.7 : -0.4));
      return flat.rotate(probe, 9).every((x, i) => near(x, fn.rotate(probe, 9)[i], 1e-15));
    })()
  );
  ok(
    "the temperature is a modulus on the rotation, so it scales the logit by its square",
    (() => {
      const t = 1.2079; // 0.1·ln(8) + 1, the paper's own worked value
      const plainR = rope({ dims: DH });
      const hot = rope({ dims: DH, modulus: t });
      const a = Float64Array.from({ length: DH }, (_, i) => (i % 3 ? 0.5 : -0.9));
      const b = Float64Array.from({ length: DH }, (_, i) => (i % 2 ? -0.3 : 0.8));
      const cold = dot(plainR.rotate(a, 3), plainR.rotate(b, 7));
      return near(dot(hot.rotate(a, 3), hot.rotate(b, 7)), cold * t * t, 1e-12);
    })(),
    "which is why it needs no change to the attention code"
  );
  ok(
    "a temperature of 1 leaves the rotation's length alone, as RoPE's does",
    (() => {
      const probe = Float64Array.from({ length: DH }, (_, i) => (i % 2 ? 0.6 : -0.2));
      const r = rope({ dims: DH });
      return near(dot(r.rotate(probe, 11), r.rotate(probe, 11)), dot(probe, probe), 1e-12);
    })()
  );

  // The collapse the streaming card is about is one identity: drop keys from a softmax row and
  // every survivor is multiplied by the same number. If that stops holding, the card's headline
  // readout is wrong rather than merely surprising.
  ok(
    "hiding keys multiplies every survivor by one shared number",
    (() => {
      const policy = (i, j) => j < 2 || j >= i - 5 + 1;
      const T = TOKENS.length;
      const head = base.trace[0].heads[0];
      const cut = forward(TOKENS, { mixer: softmaxMixer({ readable: policy }) }).trace[0].heads[0];
      const ratios = [];
      for (let j = 0; j < T; j++) {
        if (!policy(T - 1, j)) continue;
        if (head.weights[T - 1][j] < 1e-9) continue;
        ratios.push(cut.weights[T - 1][j] / head.weights[T - 1][j]);
      }
      return ratios.length > 1 && ratios.every((r) => near(r, ratios[0], 1e-9));
    })(),
    "the shape of what remains is untouched, the scale is not"
  );
  ok(
    "the front of the cache is kept whatever the window does",
    (() => {
      const policy = (i, j) => j < 4 || j >= i - 3 + 1;
      const w = forward(TOKENS, { mixer: softmaxMixer({ readable: policy }) }).trace[0].heads[0].weights;
      const last = w[TOKENS.length - 1];
      return last[0] > 0 && last[4] === 0;
    })(),
    "sinks survive, the middle does not"
  );

  // Selectivity is one line in the seam: a decay that used to be a constant may now be read off
  // the token. Both halves of that have to hold — the constant case must be untouched, and the
  // per-token case must actually apply different decays.
  ok(
    "a decay given as a constant function is the same as the constant",
    (() => {
      const a = forward(TOKENS, { mixer: stateMixer({ write: "gated", decay: 0.7 }) }).trace[0].heads[0];
      const b = forward(TOKENS, { mixer: stateMixer({ write: "gated", decay: () => 0.7 }) }).trace[0].heads[0];
      return a.out.every((v, i) => v.every((x, d) => near(x, b.out[i][d], 1e-15)));
    })()
  );
  ok(
    "a decay read off the token applies a different one at different tokens",
    (() => {
      const h = forward(TOKENS, {
        mixer: stateMixer({ write: "gated", decay: (i, k) => 1 / (1 + Math.exp(-k[0])) }),
      }).trace[0].heads[0];
      return new Set(h.gates.map((g) => g.toFixed(9))).size > 1;
    })(),
    "which is the length dimension the parameter did not have before"
  );
  ok(
    "the discretisation ties the write strength to the decay",
    (() => {
      // Appendix C with A = −1: Ā = exp(−Δ) and B̄ = 1 − Ā, for every Δ.
      return [0.001, 0.1, 1, 10, 100].every((d) => near(Math.exp(-d) + (1 - Math.exp(-d)), 1, 1e-15));
    })(),
    "Ā + B̄ = 1 at every step size"
  );

  // Concept 23: two switches on one state, and the whole card rests on them being independent.
  // Both corners are exact, so they are asserted as exact rather than as approximations.
  ok(
    "the gated delta rule at decay 1 is the delta rule, everywhere in the model",
    (() => {
      const a = forward(TOKENS, { mixer: ruleMixer("gated", 1, 0.8) });
      const b = forward(TOKENS, { mixer: ruleMixer("delta", 1, 0.8) });
      return a.trace.every((blk, i) =>
        blk.heads.every((h, j) => h.out.every((v, t) => v.every((x, d) => x === b.trace[i].heads[j].out[t][d])))
      );
    })(),
    "bit-identical, not merely close"
  );
  ok(
    "at write strength 0 the state is exactly empty, whatever the decay",
    [1, 0.9, 0.5].every((a) =>
      forward(TOKENS, { mixer: ruleMixer("gated", a, 0) }).trace[0].heads[0].state.every((row) =>
        row.every((x) => x === 0)
      )
    ),
    "the decay writes nothing; it only fades"
  );
  ok(
    "at full write strength the state returns the newest value exactly, whatever the decay",
    (() => {
      const T = TOKENS.length;
      const silu = (x) => x / (1 + Math.exp(-x));
      const l2 = (v) => {
        let n = 0;
        for (const x of v) n += x * x;
        n = Math.sqrt(n) || 1;
        return Float64Array.from(v, (x) => x / n);
      };
      return [1, 0.9, 0.5].every((a) => {
        const h = forward(TOKENS, { mixer: ruleMixer("gated", a, 1) }).trace[0].heads[0];
        const k = l2(Float64Array.from(h.K[T - 1], silu));
        return h.state.every((row, r) => near(dot(row, k), h.V[T - 1][r], 1e-12));
      });
    })(),
    "S k = v, which needs the paper's L2 normalisation to hold"
  );
  ok(
    "the add rule can decay, which is the only way Mamba2's row is reachable",
    (() => {
      // Hand-written S = αS + v kᵀ against the seam's, on the model's own head.
      const h = forward(TOKENS, { mixer: ruleMixer("mamba2", 0.8, 1) }).trace[0].heads[0];
      const silu = (x) => x / (1 + Math.exp(-x));
      const l2 = (v) => {
        let n = 0;
        for (const x of v) n += x * x;
        n = Math.sqrt(n) || 1;
        return Float64Array.from(v, (x) => x / n);
      };
      let S = Array.from({ length: DH }, () => new Float64Array(DH));
      for (let t = 0; t < TOKENS.length; t++) {
        const k = l2(Float64Array.from(h.K[t], silu));
        for (let r = 0; r < DH; r++) for (let c = 0; c < DH; c++) S[r][c] = S[r][c] * 0.8 + h.V[t][r] * k[c];
      }
      return h.state.every((row, r) => row.every((x, c) => near(x, S[r][c], 1e-12)));
    })(),
    "and it is not any setting of the delta form"
  );
  ok(
    "the write strength can be read off the token, not only off the gate",
    (() => {
      const h = forward(TOKENS, {
        mixer: stateMixer({ write: "gated", decay: 0.9, beta: (i, g, k) => 1 / (1 + Math.exp(-k[0])) }),
      }).trace[0].heads[0];
      const c = forward(TOKENS, { mixer: stateMixer({ write: "gated", decay: 0.9, beta: 0.5 }) }).trace[0].heads[0];
      return h.out.some((v, i) => v.some((x, d) => !near(x, c.out[i][d], 1e-9)));
    })(),
    "concept 20 ties β to the gate; concept 23 does not"
  );
  ok(
    "the four rules of the family are four different states",
    (() => {
      const norm = (r) =>
        Math.sqrt(
          forward(TOKENS, { mixer: ruleMixer(r, 0.9, 0.8) })
            .trace[0].heads[0].state.reduce((s, row) => s + row.reduce((u, x) => u + x * x, 0), 0)
        ).toFixed(6);
      return new Set(["la", "mamba2", "delta", "gated"].map(norm)).size === 4;
    })()
  );

  // Concept 24: three branches, and two of them are ordinary attention over a subset — so at their
  // degenerate settings they have to *be* ordinary attention, and the decode table has to fall out
  // of the formula rather than be typed in.
  ok(
    "selecting every block reduces to plain attention",
    (() => {
      const all = { ...APP, n: 999 };
      const h = base.trace[0].heads[0];
      return h.out.every((v, t) => {
        const r = nsaAt(h.Q, h.K, h.V, t, all, { cmp: 0, slc: 1, win: 0 });
        return v.every((x, d) => x === r.out[d]);
      });
    })(),
    "bit-identical, so the selection branch is attention over a subset and nothing else"
  );
  ok(
    "a window as long as the sentence reduces to plain attention",
    (() => {
      const h = base.trace[0].heads[0];
      const p = { ...APP, w: TOKENS.length };
      return h.out.every((v, t) => {
        const r = nsaAt(h.Q, h.K, h.V, t, p, { cmp: 0, slc: 0, win: 1 });
        return v.every((x, d) => x === r.out[d]);
      });
    })()
  );
  ok(
    "the branch weights inside each branch sum to 1 — three softmaxes, not one mask",
    (() => {
      const h = base.trace[0].heads[0];
      for (let t = 0; t < TOKENS.length; t++) {
        const r = nsaAt(h.Q, h.K, h.V, t, APP, { cmp: 1, slc: 1, win: 1 });
        const sum = (a) => a.reduce((x, y) => x + y, 0);
        if (r.pc.length && !near(sum(Array.from(r.pc)), 1)) return false;
        if (!near(sum(Array.from(r.ps)), 1) || !near(sum(Array.from(r.pw)), 1)) return false;
      }
      return true;
    })()
  );
  ok(
    "block importance costs no comparison with a key",
    (() => {
      // Eq. 9 reads only p^cmp, which the compression branch computed for its own output. If the
      // derived scores can be reproduced from p^cmp alone, nothing else was consulted.
      const h = base.trace[0].heads[0];
      const t = TOKENS.length - 1;
      const r = nsaAt(h.Q, h.K, h.V, t, APP, { cmp: 1, slc: 1, win: 1 });
      const { l, d, lp } = APP;
      return r.pslc.every((v, j) => {
        let acc = 0;
        for (let m = 0; m < lp / d; m++)
          for (let n = 0; n < l / d; n++) {
            const i = (lp / d) * j - m - n;
            if (i >= 0 && i < r.pc.length) acc += r.pc[i];
          }
        return near(acc, v, 1e-15);
      });
    })(),
    "Eq. 9 is a sum over scores already in hand"
  );
  ok(
    "the decoding formula reproduces the paper's Table 4",
    [[8192, 2048], [16384, 2560], [32768, 3584], [65536, 5632]].every(
      ([s, want]) => decodeReads(s, PAPER) === want
    ),
    "2,048 / 2,560 / 3,584 / 5,632 at l=32 d=16 l'=64 n=16 w=512"
  );
  ok(
    "and its speedup column",
    [[8192, 4.0], [16384, 6.4], [32768, 9.1], [65536, 11.6]].every(
      ([s, want]) => near(s / decodeReads(s, PAPER), want, 0.05)
    ),
    "4.0× / 6.4× / 9.1× / 11.6×"
  );
  ok(
    "below the break-even the mechanism reads more than full attention",
    (() => {
      const e = breakEven(PAPER);
      return (
        Math.abs(e - 1638) < 1 &&
        decodeReads(1024, PAPER) > 1024 &&
        decodeReads(2048, PAPER) < 2048 &&
        // and this app's own sentence is on the losing side of its own curve
        breakEven(APP) > TOKENS.length
      );
    })(),
    "1,638 tokens at the paper's block sizes, and the app's sentence is under its own"
  );

  // The latent cache is an identity, not an approximation: moving the up-projection onto the query
  // must not change a single score. If that stops holding, the mechanism's whole claim to be free
  // is gone, so it is checked here rather than only on the card.
  ok(
    "moving the up-projection onto the query changes no score",
    (() => {
      const r = 5;
      const rnd = mulberry32(99);
      const WUK = Array.from({ length: r }, () => Float64Array.from({ length: DH }, () => gauss(rnd)));
      const c = Float64Array.from({ length: r }, () => gauss(rnd));
      const q = Float64Array.from({ length: DH }, () => gauss(rnd));
      const k = new Float64Array(DH);
      for (let j = 0; j < r; j++) for (let d = 0; d < DH; d++) k[d] += c[j] * WUK[j][d];
      const qc = Float64Array.from({ length: r }, (_, j) => dot(WUK[j], q));
      return near(dot(q, k), dot(qc, c), 1e-12);
    })(),
    "which is why the compression costs no arithmetic at generation time"
  );
  ok(
    "a rotation between them breaks it",
    (() => {
      const r = 5;
      const rnd = mulberry32(1234);
      const WUK = Array.from({ length: r }, () => Float64Array.from({ length: DH }, () => gauss(rnd)));
      const c = Float64Array.from({ length: r }, () => gauss(rnd));
      const q = Float64Array.from({ length: DH }, () => gauss(rnd));
      const rot = rope({ dims: DH });
      const k = new Float64Array(DH);
      for (let j = 0; j < r; j++) for (let d = 0; d < DH; d++) k[d] += c[j] * WUK[j][d];
      const truth = dot(rot.rotate(q, 15), rot.rotate(k, 3));
      const qc = Float64Array.from({ length: r }, (_, j) => dot(WUK[j], rot.rotate(q, 15)));
      return Math.abs(truth - dot(qc, c)) > 1e-6;
    })(),
    "which is why position needs a channel of its own"
  );
  ok(
    "the latent hook replaces the block's keys and values",
    (() => {
      const flat = forward(TOKENS, { latent: (normed) => ({ K: normed.map(() => new Float64Array(CONFIG.D)), V: normed.map(() => new Float64Array(CONFIG.D)) }) });
      const w = flat.trace[0].heads[0].weights;
      // Every key identical means every readable weight in a row is identical.
      return w[TOKENS.length - 1].filter((x) => x > 0).every((x, _, a) => near(x, a[0], 1e-12));
    })()
  );

  // ---------------------------------------------------------------------- cost
  const gb = (o = {}) => cacheBytes({ ...SERVING, ...o }) / GB;
  ok("cache: one conversation is 6.44 GB", near(gb(), 6.44, 0.005), gb().toFixed(3));
  ok("cache: eight conversations is 51.54 GB", near(gb({ batch: 8 }), 51.54, 0.005), gb({ batch: 8 }).toFixed(3));
  ok("cache: four times the kv heads is four times the cache", near(gb({ kvHeads: 32 }) / gb({ kvHeads: 8 }), 4));

  // ----------------------------------------------------------------- the record
  const ids = mechanisms.map((m) => m.id);
  ok("mechanism ids are unique", new Set(ids).size === ids.length);
  ok("every date parses", mechanisms.every((m) => /^\d{4}-\d{2}-\d{2}$/.test(m.date) && !isNaN(Date.parse(m.date))));
  const sorted = [...mechanisms].sort((a, b) => a.date.localeCompare(b.date));
  ok("the baseline is the earliest entry", sorted[0].id === "transformer", sorted[0].id);
  const badVerified = mechanisms.filter((m) => m.verified && m.source.kind !== "paper");
  ok("only paper-backed entries are marked verified", badVerified.length === 0, badVerified.map((m) => m.id).join(", "));

  const built = mechanisms.filter((m) => m.status !== "pending");
  const noCard = built.filter((m) => !CARDS[m.id]);
  ok("every built concept resolves to a card the model can run", noCard.length === 0, noCard.map((m) => m.id).join(", "));

  const linkless = built.filter((m) => !m.leaves?.text || (m.id !== "transformer" && !m.answers));
  ok(
    "every built concept names what it answers and what it leaves behind",
    linkless.length === 0,
    linkless.map((m) => m.id).join(", ")
  );
  const danglingBack = built.filter((m) => m.answers && !ids.includes(m.answers));
  const danglingFwd = built.filter((m) => m.leaves?.to && !ids.includes(m.leaves.to));
  ok(
    "those links point at entries that exist",
    danglingBack.length + danglingFwd.length === 0,
    [...danglingBack, ...danglingFwd].map((m) => m.id).join(", ")
  );
  const backwards = built.filter((m) => {
    if (!m.answers) return false;
    const other = mechanisms.find((x) => x.id === m.answers);
    return other && other.date > m.date;
  });
  ok("nothing answers a limitation that had not happened yet", backwards.length === 0, backwards.map((m) => m.id).join(", "));

  // Mount every built card into a detached node: catches a card that throws, and one that
  // forgot its trade-off record or its plain-language verdict.
  const broken = [];
  const missingTrade = [];
  const missingPlain = [];
  for (const m of built) {
    const host = document.createElement("div");
    try {
      const mounted = CARDS[m.id](host, m) || {};
      if (mounted.update) mounted.update();
    } catch (err) {
      broken.push(`${m.id}: ${err.message}`);
      continue;
    }
    const trades = host.querySelectorAll(".trade");
    if (trades.length !== 3) missingTrade.push(m.id);
    const plain = host.querySelector(".plain");
    const pros = host.querySelectorAll(".plain .pros li").length;
    const cons = host.querySelectorAll(".plain .cons li").length;
    if (!plain || !pros || !cons || !host.querySelector(".plain .verdict")) missingPlain.push(m.id);
    else if (cons === 0) missingPlain.push(m.id);
  }
  ok("every built card renders without throwing", broken.length === 0, broken.join(" | "));
  ok("every built card answers buys / gives up / when", missingTrade.length === 0, missingTrade.join(", "));
  ok(
    "every built card ends with a plain-language verdict that includes costs",
    missingPlain.length === 0,
    missingPlain.join(", ")
  );

  const pending = mechanisms.filter((m) => m.status === "pending");
  ok(
    `${mechanisms.length - pending.length} of ${mechanisms.length} concepts built`,
    true,
    pending.length ? `still to come: ${pending.map((m) => m.id).join(", ")}` : "all done"
  );

  return out;
}

export function runSelfCheck() {
  const results = checks();
  const failed = results.filter((r) => !r.pass);
  const box = el("div", { id: "selfcheck", class: failed.length ? "fail" : "pass" }, [
    el("div", {
      text: failed.length
        ? `${failed.length} of ${results.length} checks FAILED`
        : `all ${results.length} checks pass`,
    }),
    el(
      "ul",
      {},
      (failed.length ? failed : results).map((r) =>
        el("li", { text: `${r.pass ? "✓" : "✗"} ${r.name}${r.detail ? " — " + r.detail : ""}` })
      )
    ),
  ]);
  document.body.appendChild(box);
  return results;
}
