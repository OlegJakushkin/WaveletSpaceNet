"""2-D Haar wavelets + the pyramid image tokenizer.

The current grayscale frame is turned into a **pyramid of wavelet block tokens**:
the image is split into non-overlapping tiles at a sequence of block sizes
(1024, 512, 256, 128, 64, 32 by default — coarse → fine), every tile is summarised
by its 2-D Haar coefficients, and each summary becomes one transformer token tagged
with its pyramid level and tile centre.  This is the image-side analogue of the
surface model's point tokens: a multi-scale, position-tagged token set that the
Perceiver encoder reads at a cost independent of the image resolution.

The same Haar bank (orthonormal, exactly invertible) is reused by the mesh-plane
decoder, which *emits* the 2-D Haar coefficients of the depth field and inverts
them to a depth map.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _to_pow2(img: torch.Tensor, side: int) -> torch.Tensor:
    """Resize a ``(B,1,H,W)`` grayscale image to ``(B,1,side,side)`` (area / bilinear)."""
    H, W = img.shape[-2:]
    mode = "area" if (side < H or side < W) else "bilinear"
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(img, size=(side, side), mode=mode, **kw)


class WaveletPyramidTokenizer(nn.Module):
    """Grayscale frame -> a set of multi-scale wavelet block tokens.

    For each block size ``b`` in ``levels`` the (square-resized) image is cut into a
    ``(top//b)`` × ``(top//b)`` grid of ``b×b`` tiles (``top`` = the largest level).
    Each tile is Haar-transformed once and its 4 sub-bands are adaptive-avg-pooled to
    ``pool×pool``; the flattened ``4*pool*pool`` summary is linearly projected to ``d``
    and tagged with a learned level embedding plus a Fourier encoding of the tile
    centre.  Returns ``(tokens (B, T, d), centres (B, T, 2))`` where ``centres`` are
    tile centres in ``[-1, 1]`` (used by the decoder's local attention) and
    ``T = sum_b (top//b)^2``.
    """

    def __init__(self, d: int = 256, levels=DEFAULT_LEVELS, pool: int = 4, fourier_bands: int = 6):
        super().__init__()
        self.levels = tuple(int(l) for l in levels)
        self.top = max(self.levels)
        assert all(self.top % b == 0 for b in self.levels), "every level must divide the top level"
        self.pool, self.fb, self.d = pool, fourier_bands, d
        feat = 4 * pool * pool
        self.embed = nn.Sequential(nn.Linear(feat, d), nn.LayerNorm(d))
        self.level_emb = nn.Parameter(torch.zeros(len(self.levels), d))
        self.pos = nn.Sequential(nn.Linear(2 * 2 * fourier_bands, d), nn.LayerNorm(d))
        self.register_buffer("haar", haar_filters_2d())

    def n_tokens(self) -> int:
        return sum((self.top // b) ** 2 for b in self.levels)

    def forward(self, img: torch.Tensor):
        """``img``: ``(B, 1, H, W)`` grayscale in ~[0,1] (any H, W)."""
        from .blocks import fourier_encode
        if img.dim() == 3:
            img = img[:, None]
        B = img.shape[0]
        big = _to_pow2(img, self.top)                              # (B,1,top,top)
        toks, cens = [], []
        for li, b in enumerate(self.levels):
            g = self.top // b                                      # tiles per side
            # cut into g*g tiles of b*b -> (B*g*g, 1, b, b)
            t = big.unfold(2, b, b).unfold(3, b, b)                # (B,1,g,g,b,b)
            t = t.permute(0, 2, 3, 1, 4, 5).reshape(B * g * g, 1, b, b)
            c = dwt2d(t, self.haar)                                # (B*g*g,4,b/2,b/2)
            c = F.adaptive_avg_pool2d(c, self.pool).reshape(B, g * g, -1)   # (B,g*g,4*pool*pool)
            tok = self.embed(c) + self.level_emb[li]
            # tile centres in [-1,1]
            ax = (torch.arange(g, device=img.device) + 0.5) / g * 2 - 1
            cy, cx = torch.meshgrid(ax, ax, indexing="ij")
            cen = torch.stack([cx, cy], -1).reshape(1, g * g, 2).expand(B, -1, -1)
            tok = tok + self.pos(fourier_encode(cen, self.fb))
            toks.append(tok); cens.append(cen)
        return torch.cat(toks, 1), torch.cat(cens, 1)             # (B,T,d), (B,T,2)
