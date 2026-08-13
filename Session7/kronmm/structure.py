"""
Does the Kronecker prior match each modality's structure?

The codec's entire content is "two things are similar iff they share values at the same
roles". Whether that is a good prior is not a matter of opinion -- it is a correlation between
codec cosine and the similarity that actually matters in each modality:

    text    negative edit distance between the token strings
    image   negative L2 distance between the patches, in pixel space
    audio   negative distance between log-magnitude spectra of the frames

Reported as Spearman rank correlation, which is what "does the codec order pairs the way the
modality does" means. Measured for a sweep of the filler width sigma, because sigma is the one
per-modality knob in the design: 0 for a categorical alphabet, > 0 for an ordinal one.

Real MNIST patches, synthesised audio, and real text tokens. Nothing is trained.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from mmcodec import AUDIO_FRAME, ROLE_DIM, kron_code, mu_law_encode, spectral_bytes

SIGMAS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def _cos_matrix(codes: torch.Tensor) -> np.ndarray:
    c = codes[:, :-3]
    c = c / c.norm(dim=1, keepdim=True)
    return (c @ c.T).numpy()


def _upper(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ modality samples

def image_sample(n: int, seed: int = 0):
    """Real 8x8 patches cut from MNIST digits."""
    from datasets import load_dataset
    ds = load_dataset("ylecun/mnist", split=f"train[:{max(400, n)}]")
    rng = np.random.default_rng(seed)
    patches = []
    for i in range(n):
        img = np.array(ds[int(rng.integers(0, len(ds)))]["image"], dtype=np.uint8)
        r, c = rng.integers(0, 28 - 8, 2)
        patches.append(img[r:r + 8, c:c + 8])
    P = np.stack(patches)
    truth = -np.linalg.norm(P.reshape(n, -1)[:, None, :].astype(np.float32)
                            - P.reshape(n, -1)[None, :, :].astype(np.float32), axis=-1)
    return [p.reshape(-1) for p in P], truth


def _audio_waves(n: int, seed: int, sr: int = 8000):
    """Frames of mixed tones at varied frequency, amplitude and PHASE."""
    rng = np.random.default_rng(seed)
    t = np.arange(AUDIO_FRAME) / sr
    waves, specs = [], []
    for _ in range(n):
        w = np.zeros_like(t)
        for _ in range(rng.integers(1, 4)):                 # 1-3 partials
            f0, amp = rng.uniform(100, 2000), rng.uniform(0.2, 0.9)
            w += amp * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
        w /= max(1e-9, np.abs(w).max())
        waves.append(w)
        specs.append(np.log1p(np.abs(np.fft.rfft(w * np.hanning(w.size)))[:ROLE_DIM]))
    S = np.stack(specs)
    truth = -np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
    return waves, truth


def audio_sample(n: int, seed: int = 0):
    """Frequency-bin roles: filler = quantised log-magnitude per bin."""
    waves, truth = _audio_waves(n, seed)
    return [spectral_bytes(w) for w in waves], truth


def audio_time_sample(n: int, seed: int = 0):
    """The naive time-index role, for the ablation: filler = mu-law sample."""
    waves, truth = _audio_waves(n, seed)
    return [mu_law_encode(w[:ROLE_DIM]) for w in waves], truth


def photo_sample(n: int, seed: int = 0):
    """Continuous-tone 8x8 patches: smooth gradients plus texture, i.e. mid-range values.

    MNIST is nearly binary, so the ordinal problem barely shows there. Real photographic
    content lives in the mid-range, which is where a one-hot filler hurts."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:8, 0:8] / 7.0
    patches = []
    for _ in range(n):
        a, b, c = rng.uniform(-1, 1, 3)
        base = 128 + 90 * (a * xx + b * yy + c * xx * yy)
        img = np.clip(base + rng.normal(0, 6, (8, 8)), 0, 255).astype(np.uint8)
        patches.append(img)
    P = np.stack(patches).reshape(n, -1).astype(np.float32)
    truth = -np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return [p.astype(np.uint8) for p in P], truth


def text_sample(n: int, seed: int = 0):
    """Real word tokens, with the deliberate inclusion of near-miss spellings."""
    base = ("the of and to in is it you that he was for on are with as his they at be this "
            "have from or one had by word but not what all were we when your can said there "
            "training trainer trained train compute commute nation notion run rune runs "
            "receive recieve separate seperate colour color realise realize").split()
    rng = np.random.default_rng(seed)
    toks = [base[int(i)] for i in rng.integers(0, len(base), n)]
    truth = -np.array([[edit_distance(a, b) for b in toks] for a in toks], dtype=np.float32)
    return [np.frombuffer(t.encode()[:ROLE_DIM], dtype=np.uint8) for t in toks], truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180)
    a = ap.parse_args()

    samplers = {"text": text_sample, "image": image_sample, "photo": photo_sample,
                "audio": audio_sample, "audio-time": audio_time_sample}
    print(f"Spearman correlation between codec cosine and in-modality similarity "
          f"(n={a.n} items, {a.n*(a.n-1)//2:,} pairs)\n")
    header = "modality   " + "".join(f"{'s=' + str(s):>9s}" for s in SIGMAS)
    print(header)
    print("-" * len(header))

    best = {}
    for name, sampler in samplers.items():
        values, truth = sampler(a.n)
        y = _upper(truth)
        row, scores = f"{name:10s} ", []
        for s in SIGMAS:
            mod = {"photo": "image", "audio-time": "audio"}.get(name, name)
            codes = torch.stack([kron_code(v, mod, sigma=s) for v in values])
            rho = spearmanr(_upper(_cos_matrix(codes)), y).statistic
            scores.append(rho)
            row += f"{rho:9.3f}"
        print(row)
        best[name] = SIGMAS[int(np.argmax(scores))]

    print("\nbest sigma per modality:", ", ".join(f"{k}={v}" for k, v in best.items()))
    print("text wants a categorical filler; image and audio want an ordinal one.")


if __name__ == "__main__":
    main()
