"""
Multimodal Kronecker codec — one fixed encoder, one projection, three modalities.

Session 7 assignment, Problem 2: "What is the natural extension of Kronecker, such that it
can represent images and audio as well? Yes we'll need to do some preprocessing of image and
audio patches as well, but how do we use this concept to represent all 3?"

The released codec is a Tensor Product Representation: a token is encoded as a bundle of
(filler (x) role) pairs, where the filler is a one-hot byte VALUE and the role is a one-hot
byte POSITION. Both factors are one-hot, nothing is learned, and the whole token is the sum.

Two observations turn that into a multimodal encoder.

1. THE FILLER ALPHABET IS ALREADY UNIVERSAL. All three modalities are natively 8-bit:

       text    UTF-8 byte                  0..255
       image   8-bit pixel intensity       0..255
       audio   mu-law companded sample     0..255   (the WaveNet quantization)

   mu-law is the important one -- it is not a convenience cast. It is the standard
   perceptual companding for audio, so the 256 levels are perceptually spaced rather than
   linearly spaced, which is exactly what makes a one-hot over them meaningful.

2. ONLY THE ROLE SPACE DIFFERS, and it is itself a Kronecker product where it needs to be:

       text    role = position within the token          1-D, 64 slots
       image   role = (row (x) col) within a patch       2-D, 8x8 = 64 slots
       audio   role = sample index within a frame        1-D, 64 slots

   Fixing the number of role slots at R = 64 for every modality makes the code dimension
   D = 256 x 64 = 16,384 identical across modalities, so ONE shared Linear(D, d_model)
   consumes all three. There is no per-modality embedding table anywhere in the input path.

WHY THE PRIOR IS SOUND HERE, AND WHERE IT WOULD NOT BE. The codec's whole content is
"two things are similar iff they share values at the same roles". For raw file bytes that is
a bad prior -- byte 400 of one JPEG has no stable relationship to byte 400 of another, which
is the objection to feeding compressed files to a byte model. Once the input is preprocessed
into patches the prior becomes correct: two 8x8 image patches sharing pixel values at the
same offsets really are visually similar, and two audio frames sharing mu-law amplitudes at
the same offsets really are acoustically similar. The assignment's "we'll need to do some
preprocessing of image and audio patches" is precisely the step that earns the prior.

Modality is disambiguated by a 3-dim one-hot appended to the code, so a text token and an
image patch that happened to share bytes cannot collide. Binding modality as a third
Kronecker factor would be more elegant but triples D for two bits of information.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

CHAR_DIM = 256                      # the universal 8-bit filler alphabet
ROLE_DIM = 64                       # role slots, shared by every modality
D_CODE = CHAR_DIM * ROLE_DIM        # 16,384
MODALITIES = ("text", "image", "audio")
D_TOTAL = D_CODE + len(MODALITIES)  # + modality one-hot

MU = 255.0                          # mu-law companding parameter


# ----------------------------------------------------------------- the codec core

_KERNEL_CACHE: dict[float, Tensor] = {}


def filler_kernel(sigma: float) -> Tensor:
    """
    (256, 256) matrix whose row v is the filler encoding of value v, L2-normalised.

    sigma = 0 gives the identity: the one-hot of the released codec, correct for a
    CATEGORICAL alphabet like UTF-8 bytes, where 101 and 102 have no reason to be similar.

    sigma > 0 gives a Gaussian bump over neighbouring levels, which is what an ORDINAL
    alphabet needs. Pixel intensity and mu-law amplitude are ordinal: 100 really is almost
    101. Under a one-hot they are orthogonal, so a +1 brightness shift makes a patch
    unrecognisable to the codec (measured: cosine 1.0000 -> -0.0039). The bump restores that
    locality while changing nothing else -- still fixed, still parameter-free, still
    (filler (x) role), still the same D.
    """
    if sigma not in _KERNEL_CACHE:
        if sigma <= 0:
            K = torch.eye(CHAR_DIM)
        else:
            g = torch.arange(CHAR_DIM, dtype=torch.float32)
            K = torch.exp(-(g[None, :] - g[:, None]) ** 2 / (2 * sigma ** 2))
            K = K / K.norm(dim=1, keepdim=True)
        _KERNEL_CACHE[sigma] = K
    return _KERNEL_CACHE[sigma]


# How ordinal each modality's filler alphabet is. This is the whole per-modality difference.
DEFAULT_SIGMA = {"text": 0.0, "image": 4.0, "audio": 4.0}


def kron_code(values: np.ndarray, modality: str, role_dim: int = ROLE_DIM,
              sigma: float | None = None) -> Tensor:
    """
    kappa = (1/sqrt(L)) * sum_p  filler(value_p) (x) onehot(role_p),  then z-normalised.

    `values` holds one 0..255 filler per occupied role slot, in role order. Slots beyond
    len(values) stay empty, exactly as short tokens leave trailing byte positions empty.
    """
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    v = np.asarray(values, dtype=np.int64).reshape(-1)[:role_dim]
    if v.size and (v.min() < 0 or v.max() > 255):
        raise ValueError("filler values must be bytes in 0..255")
    sigma = DEFAULT_SIGMA[modality] if sigma is None else sigma

    L = int(v.size)
    grid = torch.zeros(CHAR_DIM, role_dim, dtype=torch.float32)
    if L:
        grid[:, :L] = filler_kernel(sigma)[torch.from_numpy(v)].T / math.sqrt(L)
    out = grid.reshape(-1)
    out = (out - out.mean()) / (out.std() + 1e-6)

    tag = torch.zeros(len(MODALITIES))
    tag[MODALITIES.index(modality)] = 1.0
    return torch.cat([out, tag])


def kron_decode(code: Tensor, role_dim: int = ROLE_DIM) -> tuple[str, np.ndarray]:
    """
    Invert the codec: recover the modality and the filler values.

    Each role slot holds at most one filler, so the value is the argmax down the filler axis.
    Occupied slots sit above the z-normalised floor; empty ones are flat.
    """
    modality = MODALITIES[int(code[-len(MODALITIES):].argmax())]
    grid = code[:CHAR_DIM * role_dim].reshape(CHAR_DIM, role_dim)
    peak, arg = grid.max(dim=0)
    occupied = peak > grid.mean()
    L = int(occupied.sum())
    return modality, arg[:L].numpy().astype(np.uint8)


# ----------------------------------------------------------------- per-modality front ends

def encode_text(token: str) -> Tensor:
    """Role = byte position within the token. The released codec, at role_dim 64."""
    return kron_code(np.frombuffer(token.encode()[:ROLE_DIM], dtype=np.uint8), "text")


def encode_image(patch: np.ndarray) -> Tensor:
    """
    Role = (row (x) col) inside an 8x8 patch of 8-bit pixels, flattened row-major.

    The 2-D role is itself a Kronecker product of a row one-hot and a column one-hot; storing
    it flattened is the same vector, which is why no new machinery is needed for 2-D data.
    """
    p = np.asarray(patch, dtype=np.uint8)
    if p.shape != (8, 8):
        raise ValueError(f"expected an 8x8 patch, got {p.shape}")
    return kron_code(p.reshape(-1), "image")


def mu_law_encode(x: np.ndarray, mu: float = MU) -> np.ndarray:
    """Float waveform in [-1, 1] -> 0..255, perceptually spaced (WaveNet's quantization)."""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)      # [-1, 1]
    return np.round((y + 1) / 2 * mu).astype(np.uint8)


def mu_law_decode(q: np.ndarray, mu: float = MU) -> np.ndarray:
    y = np.asarray(q, dtype=np.float64) / mu * 2 - 1
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu


def encode_audio_time(frame: np.ndarray) -> Tensor:
    """
    The naive audio front end: role = sample index, filler = mu-law byte.

    Kept because it is the obvious first thing to try and it does not work. Two frames of the
    same tone at different phase are the same sound and have completely different samples at
    every index, so a time-index role cannot express audio similarity: measured Spearman
    against spectral distance is 0.005, i.e. nothing. See `encode_audio`.
    """
    f = np.asarray(frame, dtype=np.float64).reshape(-1)
    if f.size != ROLE_DIM:
        raise ValueError(f"expected a {ROLE_DIM}-sample frame, got {f.size}")
    return kron_code(mu_law_encode(f), "audio")


AUDIO_FRAME = 2 * ROLE_DIM          # 128 samples -> 65 rfft bins, of which we keep 64


def spectral_bytes(frame: np.ndarray) -> np.ndarray:
    """Log-magnitude spectrum of a frame, quantised to 8 bits, one byte per frequency bin."""
    f = np.asarray(frame, dtype=np.float64).reshape(-1)
    if f.size != AUDIO_FRAME:
        raise ValueError(f"expected a {AUDIO_FRAME}-sample frame, got {f.size}")
    mag = np.abs(np.fft.rfft(f * np.hanning(f.size)))[:ROLE_DIM]
    log = np.log1p(mag)
    top = log.max()
    if top <= 0:
        return np.zeros(ROLE_DIM, dtype=np.uint8)
    return np.round(log / top * 255).astype(np.uint8)


def encode_audio(frame: np.ndarray) -> Tensor:
    """
    Role = FREQUENCY BIN, filler = quantised log-magnitude.

    This is the role space audio actually wants. The rule the three modalities share is that
    the role must be the axis along which the modality is stable: byte position for text,
    pixel position for an image patch, frequency for a sound. Phase, which the time-index
    version was drowning in, is discarded here -- deliberately, since it is the thing audio
    similarity is invariant to.
    """
    return kron_code(spectral_bytes(frame), "audio")


def demo() -> None:
    """Round-trip every modality through one codec, and show the shared projection."""
    print(f"char_dim={CHAR_DIM}  role_dim={ROLE_DIM}  D={D_CODE}  (+{len(MODALITIES)} tag) "
          f"= {D_TOTAL}")

    # text
    for tok in ["the", "training", "भारत", "🙂"]:
        m, vals = kron_decode(encode_text(tok))
        assert m == "text" and bytes(vals) == tok.encode()[:ROLE_DIM], tok
    print("text   round-trip exact (ascii, devanagari, emoji)")

    # image
    rng = np.random.default_rng(0)
    for _ in range(50):
        patch = rng.integers(0, 256, (8, 8), dtype=np.uint8)
        m, vals = kron_decode(encode_image(patch))
        assert m == "image" and np.array_equal(vals.reshape(8, 8), patch)
    print("image  round-trip exact on 50 random 8x8 patches")

    # audio, time-domain front end: the codec recovers the mu-law bytes exactly
    t_time = np.arange(ROLE_DIM) / 8000.0
    for f0 in (110.0, 440.0, 1760.0):
        wave = 0.8 * np.sin(2 * np.pi * f0 * t_time)
        m, vals = kron_decode(encode_audio_time(wave))
        assert m == "audio" and np.array_equal(vals, mu_law_encode(wave))
        err = np.abs(mu_law_decode(vals) - wave).max()
        assert err < 0.02, err
    print(f"audio  time-domain round-trip exact, max mu-law reconstruction error {err:.4f}")

    # audio, spectral front end: recovers the quantised spectrum, and is far less
    # phase-sensitive than the time-domain one -- which is why it is the right role space.
    # Not phase-INVARIANT: a 128-sample frame holds under two periods at 110 Hz, so windowed
    # spectral leakage leaves real phase dependence at low frequencies. Measured, not assumed.
    t_frame = np.arange(AUDIO_FRAME) / 8000.0
    spec_drift, time_drift = [], []
    for f0 in (110.0, 440.0, 1760.0):
        wave = 0.8 * np.sin(2 * np.pi * f0 * t_frame)
        m, vals = kron_decode(encode_audio(wave))
        assert m == "audio" and np.array_equal(vals, spectral_bytes(wave))
        shifted = 0.8 * np.sin(2 * np.pi * f0 * t_frame + 1.1)      # same sound, new phase
        spec_drift.append(np.abs(spectral_bytes(shifted).astype(int) - vals.astype(int)).mean())
        time_drift.append(np.abs(mu_law_encode(shifted[:ROLE_DIM]).astype(int)
                                 - mu_law_encode(wave[:ROLE_DIM]).astype(int)).mean())
    print(f"audio  spectral round-trip exact; mean code drift under a phase shift: "
          f"{np.mean(spec_drift):.1f}/255 spectral vs {np.mean(time_drift):.1f}/255 time-domain")

    # the ordinal problem, and the fix
    rng2 = np.random.default_rng(0)
    p = rng2.integers(20, 236, (8, 8), dtype=np.uint8)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a[:-3], b[:-3], dim=0))
    print("\nbrightness shift on one patch     sigma=0 (one-hot)  sigma=4 (ordinal)")
    for d in (1, 2, 4, 8, 16, 32):
        q = (p.astype(int) + d).clip(0, 255).astype(np.uint8)
        c0 = cos(kron_code(p.reshape(-1), "image", sigma=0.0),
                 kron_code(q.reshape(-1), "image", sigma=0.0))
        c4 = cos(kron_code(p.reshape(-1), "image", sigma=4.0),
                 kron_code(q.reshape(-1), "image", sigma=4.0))
        print(f"  +{d:<4d}{c0:+30.4f} {c4:+17.4f}")
    print("  a one-hot filler makes a +1 shift as distant as a +32 one; the bump does not")

    # one projection consumes all three
    proj = torch.nn.Linear(D_TOTAL, 512, bias=False)
    batch = torch.stack([encode_text("hello"),
                         encode_image(rng.integers(0, 256, (8, 8), dtype=np.uint8)),
                         encode_audio(np.sin(2 * np.pi * 440 * t_frame))])
    out = proj(batch)
    print(f"\none shared Linear({D_TOTAL}, 512): {batch.shape} -> {out.shape}, "
          f"{sum(p.numel() for p in proj.parameters())/1e6:.2f}M params total")
    print("no per-modality embedding table anywhere in the input path")


if __name__ == "__main__":
    demo()
