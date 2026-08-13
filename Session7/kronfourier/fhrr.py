"""
Each character as a Fourier wave, added to make a word.

    "What is a REAL Fourier alternative of Kronecker? Why can't I represent each character
     like a Fourier wave, and just add them to make a word!"     -- Session 7, problem 4

Taken literally, a sum of per-character waves is order-blind: "abc" and "cba" give the same
vector. Position has to enter through *phase*, and once it does the construction has a name --
Fourier Holographic Reduced Representations.

    phi[v]      unit-modulus phasor per byte value, fixed random, N complex dims
    psi^p       position p as a phase rotation exp(i*p*theta), theta fixed random
    encode(b)   (1/sqrt(L)) * sum_p phi[b_p] (*) psi^p           elementwise complex multiply

Stored as [real, imag], so D = 2N real dimensions and N = 4096 matches the released codec's
D = 8192 exactly. Nothing is learned; phi and theta come from a seed.

Three properties the byte codec cannot have:

    length-free     D does not depend on L. No pos_dim, no 32-byte crop.
    unbinding       z (*) conj(psi^p) ~= phi[b_p] + noise, so bytes come back out
    shift-equivariant   prepending k bytes multiplies the WHOLE code by psi^k, which is one
                        elementwise multiply on the code -- no access to the bytes needed
"""

from __future__ import annotations

import torch

CHAR_DIM = 256
N_DEFAULT = 4096                  # complex dims -> D = 8192 real, matching the released codec
SEED = 0


def tables(N: int = N_DEFAULT, seed: int = SEED, device="cpu"):
    """(phi[256, N], theta[N]) -- fixed, parameter-free, reproducible from the seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    phase = torch.rand(CHAR_DIM, N, generator=g) * 2 * torch.pi
    theta = torch.rand(N, generator=g) * 2 * torch.pi
    return torch.polar(torch.ones_like(phase), phase).to(device), theta.to(device)


def psi(theta: torch.Tensor, p) -> torch.Tensor:
    """Position p as a phasor. p may be fractional or negative -- the encoding is defined for
    every real p, which is the point of fractional power encoding."""
    p = torch.as_tensor(p, dtype=theta.dtype, device=theta.device)
    ang = p.reshape(-1, 1) * theta
    return torch.polar(torch.ones_like(ang), ang).squeeze(0)


def encode(byte_seq: bytes, phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """A word is the sum of its character waves, each rotated by its position."""
    L = len(byte_seq)
    if L == 0:
        return torch.zeros(theta.numel(), dtype=torch.complex64, device=theta.device)
    idx = torch.tensor(list(byte_seq), dtype=torch.long, device=theta.device)
    pos = psi(theta, torch.arange(L, dtype=theta.dtype, device=theta.device))
    return (phi[idx] * pos).sum(0) / L ** 0.5


def shift(z: torch.Tensor, k, theta: torch.Tensor) -> torch.Tensor:
    """Move every character k positions along, working on the code alone."""
    return z * psi(theta, k)


def unbind(z: torch.Tensor, p, phi: torch.Tensor, theta: torch.Tensor) -> int:
    """Which byte sits at position p? Rotate the position out, then match against phi.

    The match is Re<y, conj(phi_v)> summed over dims, i.e. the real inner product on the
    [real, imag] representation -- the same quantity a linear readout would compute."""
    y = z * psi(theta, -torch.as_tensor(p, dtype=theta.dtype, device=theta.device))
    return int((phi.conj() * y).sum(-1).real.argmax())


def decode(z: torch.Tensor, L: int, phi: torch.Tensor, theta: torch.Tensor) -> bytes:
    """Unbind every position. L is needed: the code carries no length marker."""
    return bytes(unbind(z, p, phi, theta) for p in range(L))


def as_real(z: torch.Tensor) -> torch.Tensor:
    """[real, imag] -- the D = 2N vector a projection would actually consume."""
    return torch.cat([z.real, z.imag], dim=-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = as_real(a), as_real(b)
    return float(x @ y / (x.norm() * y.norm() + 1e-12))


def aligned_cosine(a: torch.Tensor, b: torch.Tensor, theta: torch.Tensor, span: int = 12):
    """max over shifts of cos(a, shift(b, k)) -- the similarity the byte codec cannot express.

    Costs one elementwise multiply per candidate shift and never touches the token text, so it
    is available to anything holding the code, including a model's hidden state."""
    best = (-1.0, 0)
    for k in range(-span, span + 1):
        c = cosine(a, shift(b, k, theta))
        if c > best[0]:
            best = (c, k)
    return best


def _demo():
    phi, theta = tables()

    # 1. shift equivariance -- the property the whole submission rests on
    b = b"ation"
    z = encode(b, phi, theta)
    direct = (phi[torch.tensor(list(b))] * psi(theta, torch.arange(2, 2 + len(b),
                                                                  dtype=theta.dtype))
              ).sum(0) / len(b) ** 0.5
    assert torch.allclose(shift(z, 2, theta), direct, atol=1e-5), "shift is not equivariant"

    # 2. unbinding recovers a short token exactly
    for w in [b"a", b"train", b"training", "भारत".encode()]:
        assert decode(encode(w, phi, theta), len(w), phi, theta) == w, w

    # 3. D does not depend on length -- no crop, no pos_dim
    short, long = encode(b"a", phi, theta), encode(b"x" * 200, phi, theta)
    assert short.shape == long.shape == (N_DEFAULT,)

    # 4. the P3 failure, gone: these are bit-identical under the released codec at pos_dim=32
    a = encode("अंतर्राष्ट्रीयकरण".encode(), phi, theta)
    c = encode("अंतर्राष्ट्रीयता".encode(), phi, theta)
    assert not torch.allclose(a, c), "long tokens still collide"

    print(f"fhrr: ok  (N={N_DEFAULT}, D={2*N_DEFAULT})")
    print(f"  shift equivariant, unbind exact on short tokens, length-free")
    print(f"  the lesson's colliding pair now at cosine {cosine(a, c):.4f}")


if __name__ == "__main__":
    _demo()
