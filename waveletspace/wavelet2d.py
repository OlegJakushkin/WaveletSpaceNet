"""2-D Haar wavelets + multi-level analysis / synthesis for the mesh-plane decoder.

The depth field is still represented and *emitted* in the Haar wavelet domain — the
project's thesis — but the decoder now works **multi-level** instead of a single octave.
A depth map of side ``P`` is synthesised coarse-to-fine from a low-resolution
approximation ``LL0`` (side ``P/2**J``) plus ``J`` detail triplets ``(LH, HL, HH)`` at
sides ``P/2**J, …, P/2``.  Each ``idwt2d`` doubles the resolution, so a thin object gets
its own detail coefficients at *several* scales instead of being baked into one coarse
cell.  The same bank decomposes a ground-truth depth map into matching targets for the
multi-level wavelet loss.

Image *tokenisation* used to live here too; it now lives in
:mod:`waveletspace.encoder` (a learned conv-FPN), because average-pooling signed Haar
detail bands destroyed exactly the small-surface information we need.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Kept for checkpoint-config back-compat (older cfgs may carry a ``levels`` field).
DEFAULT_LEVELS = (1024, 512, 256, 128, 64, 32)


def haar_filters_2d(device=None, dtype=torch.float32) -> torch.Tensor:
    """The 4 separable 2-D Haar filters as a ``(4, 1, 2, 2)`` conv weight.

    Order is ``LL`` (coarse approximation) then the three detail bands ``LH, HL, HH``.
    The bank is orthonormal, so :func:`idwt2d` synthesizes the signal back exactly.
    """
    l = torch.tensor([1.0, 1.0], dtype=dtype) / np.sqrt(2.0)
    h = torch.tensor([1.0, -1.0], dtype=dtype) / np.sqrt(2.0)
    banks = (l, h)
    filt = [torch.einsum("i,j->ij", banks[a], banks[b]) for a in range(2) for b in range(2)]
    return torch.stack(filt, 0).unsqueeze(1).to(device=device)            # (4,1,2,2)


def dwt2d(x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """1-level 2-D DWT: ``(B, 1, H, W) -> (B, 4, H/2, W/2)`` (even dims)."""
    if w is None:
        w = haar_filters_2d(x.device, x.dtype)
    return F.conv2d(x, w, stride=2)


def idwt2d(c: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse 1-level 2-D DWT: ``(B, 4, h, w) -> (B, 1, 2h, 2w)`` (exact)."""
    if w is None:
        w = haar_filters_2d(c.device, c.dtype)
    return F.conv_transpose2d(c, w, stride=2)


def haar_analysis(x: torch.Tensor, levels: int, w: torch.Tensor | None = None):
    """Decompose ``(B,1,P,P)`` into ``(LL0 (B,1,g0,g0), dets)`` where ``dets`` is a list,
    coarsest-first, of detail triplets ``(B,3,s,s)`` at sides ``g0, 2*g0, …, P/2``.

    Mirrors :func:`haar_synthesis`, so the depth GT can be turned into the exact targets
    the decoder emits.
    """
    if w is None:
        w = haar_filters_2d(x.device, x.dtype)
    a = x
    dets = []
    for _ in range(levels):
        c = dwt2d(a, w)                       # (B,4,h/2,w/2)
        a = c[:, :1]                          # LL approximation
        dets.append(c[:, 1:])                 # (B,3,...)  fine -> coarse as we go down
    return a, dets[::-1]                       # LL0, dets coarsest-first


def haar_synthesis(ll0: torch.Tensor, dets, w: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse of :func:`haar_analysis`: ``LL0`` + coarsest-first ``dets`` -> ``(B,1,P,P)``."""
    if w is None:
        w = haar_filters_2d(ll0.device, ll0.dtype)
    a = ll0
    for det in dets:                          # coarsest -> finest
        a = idwt2d(torch.cat([a, det], 1), w)
    return a
