"""
"Same embedding gives the same Kronecker" -- what actually has to be inverted.

    "Kronecker is forward deterministic (same word always gives the same embedding). How do I
     make a reverse of this? If we can do this, then we can get rid of the final head as well!
     Then we can have a vocab of 1M as well without any issues."   -- Session 7, problem 5

The codec is not the obstacle. Reshape D -> (256, pos_dim), argmax down the byte axis per
column, and the bytes come back exactly -- z-normalisation is affine and does not move an
argmax. That is a solved problem before this file starts.

The obstacle is `W_proj`, the one trainable input-side tensor, mapping D=8192 down to d_model.
It is not injective, so "invert W_proj" is, read literally, impossible. This file is about why
that reading is the wrong one:

    kappa is SPARSE. Before z-normalisation the grid holds at most `pos_dim` non-zeros out of
    8192, one per byte position. Recovering a sparse vector from few linear measurements is
    not inversion, it is compressed sensing, and it does not need W to be invertible or even
    square. It needs d to be large enough relative to the sparsity, which is `pos_dim`, NOT
    relative to D and NOT relative to the vocabulary.

So the question this file answers is: how small can d_model be before a token stops coming
back? Everything here is exact arithmetic on real vocabularies. Nothing is trained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kronbudget"))
from roles import CHAR_DIM, decode, encode                          # noqa: E402

POS_DIM = 32
D = CHAR_DIM * POS_DIM                                              # 8192, as released


def projection(d: int, kind: str = "orth", seed: int = 0, D_in: int = D) -> torch.Tensor:
    """[d, D_in]. `orth` gives orthonormal ROWS, so W @ W.T == I_d and the pseudo-inverse is
    exactly W.T -- the cheapest possible decoder, one matmul, no factorisation."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d, D_in, generator=g) / D_in ** 0.5
    if kind == "orth":
        W = torch.linalg.qr(W.T, mode="reduced")[0].T.contiguous()
    return W


def forward(kappa: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """What the released module does: one matmul, D -> d_model."""
    return kappa @ W.T


def backward(h: torch.Tensor, W: torch.Tensor, kind: str = "orth") -> torch.Tensor:
    """h -> an estimate of kappa. For orthonormal rows this is the exact pseudo-inverse."""
    if kind == "orth":
        return h @ W
    return torch.linalg.lstsq(W, h.T).solution.T if h.ndim > 1 else torch.linalg.lstsq(
        W, h.unsqueeze(-1)).solution.squeeze(-1)


def infer_length(kappa_hat: torch.Tensor) -> int:
    """How long was the token? The code carries no length marker.

    With an exact kappa this never comes up: unoccupied columns sit exactly at the
    z-normalisation floor and the released decoder just walks until the first gap. After a
    lossy projection every column has energy, so the gap has to be found rather than seen.
    Occupied columns are contiguous from 0, so the length is the largest consecutive drop in
    per-column peak height."""
    s = kappa_hat.reshape(CHAR_DIM, POS_DIM).max(dim=0).values
    return int((s[:-1] - s[1:]).argmax()) + 1


def decode_est(kappa_hat: torch.Tensor, L: int | None = None) -> bytes:
    """Argmax down the byte axis, for L columns. L=None infers it."""
    if L is None:
        L = infer_length(kappa_hat)
    grid = kappa_hat.reshape(CHAR_DIM, POS_DIM)
    return bytes(int(grid[:, p].argmax()) for p in range(min(L, POS_DIM)))


def roundtrip(byte_seq: bytes, W: torch.Tensor, kind="orth", noise=0.0, seed=0,
              known_length=True) -> bytes:
    """bytes -> kappa -> h -> kappa_hat -> bytes. The whole claim, one call.

    `known_length=True` isolates the compressed-sensing question (are the BYTES recoverable);
    False is the deployable question, where the length has to be recovered too."""
    k = encode(byte_seq, POS_DIM, "trunc")
    h = forward(k, W)
    if noise:
        g = torch.Generator().manual_seed(seed)
        h = h + noise * h.norm() / h.numel() ** 0.5 * torch.randn(h.shape, generator=g)
    kh = backward(h, W, kind)
    return decode_est(kh, min(len(byte_seq), POS_DIM) if known_length else None)


def _demo():
    words = [b"a", b"train", b"training", "भारत".encode(), b"kronecker"]

    # 1. the codec alone is exactly invertible -- this is the part that was never the problem
    for w in words:
        assert decode(encode(w, POS_DIM, "trunc"), POS_DIM) == w, w

    # 2. through a projection that is not injective at all: 8192 -> 512, a 16x reduction
    W = projection(512)
    assert torch.allclose(W @ W.T, torch.eye(512), atol=1e-4), "rows are not orthonormal"
    for w in words:
        assert roundtrip(w, W) == w, (w, roundtrip(w, W))

    # 3. blind decoding has to find the length too, and that is a separate failure
    W = projection(512)
    blind = [roundtrip(w, W, known_length=False) == w for w in words]

    # 4. it keeps working well below d=512, because kappa is sparse, not because W is square
    W192 = projection(192)
    got = [roundtrip(w, W192) == w for w in words]
    print("kroninv: ok  (codec exact; d=512 recovers every word through a 16x reduction)")
    print(f"  d=512, length known : {len(words)}/{len(words)}")
    print(f"  d=512, length blind : {sum(blind)}/{len(words)}")
    print(f"  d=192, length known : {sum(got)}/{len(words)}  (43x reduction)")


if __name__ == "__main__":
    _demo()
