"""
Does one shared input path actually work across three modalities?

Task      a sequence of 4 items drawn from one source -> classify the source (28 classes:
          10 MNIST digits, 8 pitch classes, 10 word classes). Mixed batches: any sequence
          may be text, image or audio, and the model is told nothing about which.

Arms      shared    the unified Kronecker code -> ONE Linear(16387, d) for all three
          per-mod   three separate learned adapters, the ordinary way to do this:
                    Linear(64, d) on raw pixels, Linear(64, d) on spectrum bytes,
                    Embedding(V, d) on word ids

The honest framing. `per-mod` is the *smaller* arm here, and that is the point worth being
straight about: the released codec's parameter win comes from replacing a V x d_model
vocabulary table, and images and audio have no vocabulary to replace. A raw 8x8 patch is 64
numbers, so a direct linear on it is far cheaper than a 16,384-dim code. For non-text
modalities the Kronecker path buys unification and vocabulary-freeness, not parameter savings.
So the question this experiment asks is not "is it cheaper" but "what does unification cost".
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcodec import AUDIO_FRAME, D_TOTAL, ROLE_DIM, kron_code, spectral_bytes

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 4
N_DIGIT, N_PITCH, N_WORD = 10, 8, 10
N_CLASS = N_DIGIT + N_PITCH + N_WORD
WORDS = ["the", "training", "compute", "nation", "receive",
         "separate", "colour", "language", "kronecker", "embedding"]


# ---------------------------------------------------------------- data

def build_pool(n_per_class=64, seed=0):
    """Return per-class pools of raw items, one entry per (modality, class)."""
    rng = np.random.default_rng(seed)
    pool = {}

    from datasets import load_dataset
    ds = load_dataset("ylecun/mnist", split="train[:6000]")
    labels = np.array(ds["label"])
    for d in range(N_DIGIT):
        idx = np.flatnonzero(labels == d)[:n_per_class]
        items = []
        for i in idx:
            img = np.array(ds[int(i)]["image"], dtype=np.uint8)
            r, c = rng.integers(4, 28 - 12, 2)
            items.append((img[r:r + 8, c:c + 8].reshape(-1), ROLE_DIM))
        pool[("image", d)] = items

    t = np.arange(AUDIO_FRAME) / 8000.0
    for k in range(N_PITCH):
        f0 = 110.0 * (2 ** (k / 4.0))                     # 8 pitch classes over 2 octaves
        items = []
        for _ in range(n_per_class):
            w = np.zeros_like(t)
            for h in (1, 2, 3):                            # harmonics, random amplitude/phase
                w += rng.uniform(0.2, 1.0) / h * np.sin(
                    2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
            w *= rng.uniform(0.3, 1.0) / max(1e-9, np.abs(w).max())
            items.append((spectral_bytes(w), ROLE_DIM))
        pool[("audio", k)] = items

    for w_i, word in enumerate(WORDS):
        raw = np.frombuffer(word.encode()[:ROLE_DIM], dtype=np.uint8)
        padded = np.zeros(ROLE_DIM, dtype=np.uint8); padded[:raw.size] = raw
        # items are (padded bytes, true length): text is variable-length, patches are not
        pool[("text", w_i)] = [(padded, int(raw.size))] * n_per_class
    return pool


def class_id(modality, k):
    return {"image": 0, "audio": N_DIGIT, "text": N_DIGIT + N_PITCH}[modality] + k


def make_batch(pool, keys, bs, rng):
    """Each example: SEQ items from one (modality, class) pool."""
    xs, ls, ys, mods = [], [], [], []
    for _ in range(bs):
        modality, k = keys[rng.integers(0, len(keys))]
        items = pool[(modality, k)]
        pick = [items[int(rng.integers(0, len(items)))] for _ in range(SEQ)]
        xs.append(np.stack([q[0] for q in pick]))
        ls.append(np.array([q[1] for q in pick]))
        ys.append(class_id(modality, k)); mods.append(modality)
    return np.stack(xs), np.stack(ls), np.array(ys), mods


# ---------------------------------------------------------------- input paths

class SharedKron(nn.Module):
    """One fixed codec + one projection, for every modality."""

    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(D_TOTAL, d_model, bias=False)

    def forward(self, raw, lens, mods):
        codes = torch.stack([
            torch.stack([kron_code(raw[b, s, :lens[b, s]], mods[b])
                         for s in range(raw.shape[1])])
            for b in range(raw.shape[0])])
        return self.projection(codes.to(DEV))


class PerModality(nn.Module):
    """
    The ordinary approach: a separate learned adapter per modality.

    `hidden` gives each continuous adapter a hidden layer. Without it the image adapter is a
    bare linear on raw pixels, which is a strawman -- the Kronecker code is a large nonlinear
    expansion, so the baseline should at least be allowed one nonlinearity.
    """

    def __init__(self, d_model, hidden=0):
        super().__init__()
        def adapter():
            if hidden:
                return nn.Sequential(nn.Linear(ROLE_DIM, hidden), nn.GELU(),
                                     nn.Linear(hidden, d_model, bias=False))
            return nn.Linear(ROLE_DIM, d_model, bias=False)
        self.image, self.audio = adapter(), adapter()
        self.text = nn.Embedding(len(WORDS), d_model)
        self.out_dim = d_model

    def forward(self, raw, lens, mods):
        x = torch.from_numpy(raw).float().to(DEV) / 255.0
        out = torch.zeros(raw.shape[0], raw.shape[1], self.out_dim, device=DEV)
        for b, m in enumerate(mods):
            if m == "image":
                out[b] = self.image(x[b])
            elif m == "audio":
                out[b] = self.audio(x[b])
            else:
                wid = WORDS.index(bytes(raw[b, 0, :lens[b, 0]]).decode())
                out[b] = self.text(torch.full((raw.shape[1],), wid, device=DEV))
        return out


class Net(nn.Module):
    def __init__(self, front, d_model=128, layers=2, heads=4):
        super().__init__()
        self.front = front
        self.pos = nn.Parameter(torch.zeros(1, SEQ, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_CLASS)

    def forward(self, raw, lens, mods):
        x = self.front(raw, lens, mods) + self.pos
        return self.head(self.norm(self.body(x)).mean(1))


def run(arm, pool, steps, seed=0, d_model=128, bs=64, lr=3e-4, quiet=False):
    torch.manual_seed(seed)
    keys = sorted(pool)
    rng = np.random.default_rng(seed + 1)
    front = {"shared": lambda: SharedKron(d_model),
             "per-mod": lambda: PerModality(d_model),
             "per-mod-mlp": lambda: PerModality(d_model, hidden=512)}[arm]()
    model = Net(front, d_model).to(DEV)
    n_in = sum(p.numel() for p in model.front.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for _ in range(steps):
        x, l, y, m = make_batch(pool, keys, bs, rng)
        loss = F.cross_entropy(model(x, l, m), torch.from_numpy(y).to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    ev = np.random.default_rng(1234)
    per, tot, hit = {}, 0, 0
    with torch.no_grad():
        for _ in range(20):
            x, l, y, m = make_batch(pool, keys, bs, ev)
            ok = (model(x, l, m).argmax(-1).cpu().numpy() == y)
            for i, mm in enumerate(m):
                a, b = per.get(mm, (0, 0))
                per[mm] = (a + int(ok[i]), b + 1)
            hit += ok.sum(); tot += len(ok)
    acc = hit / tot
    if not quiet:
        detail = "  ".join(f"{k}:{v[0]/v[1]:.3f}" for k, v in sorted(per.items()))
        print(f"  {arm:8s} input params={n_in/1e6:.2f}M  acc={acc:.3f}   {detail}")
    return acc, n_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--arms", nargs="+", default=["shared", "per-mod", "per-mod-mlp"])
    a = ap.parse_args()
    pool = build_pool()
    print(f"device={DEV}  {N_CLASS} classes ({N_DIGIT} digit / {N_PITCH} pitch / "
          f"{N_WORD} word), seq={SEQ}, mixed batches\n")
    for arm in a.arms:
        run(arm, pool, a.steps)


if __name__ == "__main__":
    main()
