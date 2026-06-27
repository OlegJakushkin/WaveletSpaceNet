"""Shared transformer / positional primitives for WaveletSpaceNet.

These are the same Perceiver building blocks used by the *Points-as-(Super)Tori*
``PerceiverWaveNet`` (waveshape/wavelet.py) — a small set of attention blocks,
a Fourier positional encoder and farthest-point sampling — lifted here so this
repo is self-contained (numpy + torch only).  WaveletSpaceNet reuses them to fuse
a ``[context | SEP | image-pyramid]`` token sequence exactly as the surface model
fuses ``[context | SEP | main]`` point tokens.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fourier_encode(x: torch.Tensor, bands: int = 8) -> torch.Tensor:
    """``(..., C)`` in ~[-1, 1] -> ``(..., C*2*bands)`` sinusoidal positional features."""
    freqs = (2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)) * np.pi
    xb = x[..., None] * freqs
    return torch.cat([torch.sin(xb), torch.cos(xb)], -1).flatten(-2)


def fps(x: torch.Tensor, n: int) -> torch.Tensor:
    """Farthest-point sampling: ``x (B, N, D)`` -> indices ``(B, n)`` covering the set.

    Falls back to all indices (padded by repeating the last) when ``N <= n`` so the
    caller always gets exactly ``n`` indices regardless of how many points it has.
    """
    B, N = x.shape[0], x.shape[1]
    if N == 0:
        return torch.zeros(B, n, dtype=torch.long, device=x.device)
    if N <= n:
        idx = torch.arange(N, device=x.device)
        idx = torch.cat([idx, idx[-1:].expand(n - N)]) if n > N else idx
        return idx[None].expand(B, -1).contiguous()
    idx = torch.zeros(B, n, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), 1e10, device=x.device)
    far = torch.zeros(B, dtype=torch.long, device=x.device)
    ar = torch.arange(B, device=x.device)
    for i in range(n):
        idx[:, i] = far
        d = ((x - x[ar, far][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(1)
    return idx


class MHA(nn.Module):
    """Plain multi-head cross/self attention (query ``x`` attends to ``ctx``)."""

    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)

    def forward(self, x, ctx):
        B, Nq, D = x.shape; Nk = ctx.shape[1]
        q = self.q(x).view(B, Nq, self.h, self.dk).transpose(1, 2)
        k = self.k(ctx).view(B, Nk, self.h, self.dk).transpose(1, 2)
        v = self.v(ctx).view(B, Nk, self.h, self.dk).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.dk), -1)
        return self.o((a @ v).transpose(1, 2).reshape(B, Nq, D))


class CrossBlock(nn.Module):
    """Pre-norm cross-attention + MLP (the Perceiver encoder/decoder unit)."""

    def __init__(self, d: int, heads: int = 8, mlp: int = 4):
        super().__init__()
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d); self.att = MHA(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(), nn.Linear(d * mlp, d))

    def forward(self, x, ctx):
        x = x + self.att(self.nq(x), self.nk(ctx))
        return x + self.ff(self.n2(x))


class SelfBlock(nn.Module):
    """Pre-norm self-attention + MLP (latent refinement)."""

    def __init__(self, d: int, heads: int = 8, mlp: int = 4):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.att = MHA(d, heads); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(), nn.Linear(d * mlp, d))

    def forward(self, x):
        h = self.n1(x); x = x + self.att(h, h)
        return x + self.ff(self.n2(x))


class LocalAttn(nn.Module):
    """Per-query attention over each query's own ``k`` neighbour tokens ``(B, Q, k, d)``.

    The decoder reads detail for each mesh-plane query pixel from its nearest image
    tokens — the 2-D analogue of the surface model reading detail bands from a query
    point's nearest cloud tokens.
    """

    def __init__(self, d: int, heads: int = 8, mlp: int = 4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(), nn.Linear(d * mlp, d))

    def forward(self, x, nbr):                              # x:(B,Q,d) nbr:(B,Q,k,d)
        B, Q, k, D = nbr.shape
        qn, kn = self.nq(x), self.nk(nbr)
        q = self.q(qn).view(B, Q, self.h, self.dk)
        kk = self.k(kn).view(B, Q, k, self.h, self.dk)
        vv = self.v(kn).view(B, Q, k, self.h, self.dk)
        a = torch.softmax((q[:, :, None] * kk).sum(-1) / np.sqrt(self.dk), 2)   # (B,Q,k,h)
        out = (a[..., None] * vv).sum(2).reshape(B, Q, D)
        x = x + self.o(out)
        return x + self.ff(self.n2(x))
