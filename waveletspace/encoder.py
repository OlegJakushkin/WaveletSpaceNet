"""Learned multi-scale image embedding — the ``<=10M`` preprocessing module.

The original :class:`waveletspace.wavelet2d.WaveletPyramidTokenizer` summarised each
image tile by *average-pooling its Haar coefficients*.  That op averages the **signed**
detail bands of a thin object (a chair leg, a table edge) to ~zero, so the token it
emits can no longer tell that the tile contained a small surface — the dominant reason
the network only ever predicted smooth planes.  It also used **25.6 k** parameters out
of a 10 M budget.

``EdgeFPNTokenizer`` replaces that with a small *learned* convolutional encoder:

    grayscale frame -> [gray | sobel-magnitude | local-variance]   (fixed, 0-param adapter)
                    -> ConvNeXt-lite stem + 4 stages (strides 4/8/16/32)
                    -> FPN (laterals + top-down + smooth)
                    -> per-cell tokens (projected to d, tagged with level + centre)

The grayscale-only input (plus *computed* edge channels) keeps the model sensor-agnostic
— the same path works for IR / night / underwater frames where colour is unavailable.

The encoder is fed the coarse levels (strides 8/16/32) as a bounded token set; the finest
**stride-4 feature map** is returned separately and handed to the mesh-plane decoder's
*windowed* local gather, so chair-leg-scale evidence reaches the depth head without
inflating the Perceiver's cross-attention cost.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import fourier_encode


# --------------------------------------------------------------------------- #
#  Fixed (0-param) grayscale -> 3-channel edge adapter
# --------------------------------------------------------------------------- #
class EdgeChannels(nn.Module):
    """``(B,1,H,W)`` grayscale -> ``(B,3,H,W)`` = [gray, sobel-magnitude, local-variance].

    All kernels are fixed buffers, so this adds **no parameters** and works on any
    single-channel modality (visible / IR / night / sonar).  The two computed channels
    make object boundaries explicit *before* the learned stem, which matters because the
    input is a noised, partly-dropped-out splat render where faint edges are easily lost.
    """

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[1.0, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4.0
        sy = sx.t().contiguous()
        self.register_buffer("sobel", torch.stack([sx, sy])[:, None])   # (2,1,3,3)
        self.register_buffer("box", torch.ones(1, 1, 3, 3) / 9.0)       # 3x3 mean

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img[:, None]
        g = F.conv2d(F.pad(img, (1, 1, 1, 1), mode="replicate"), self.sobel)   # (B,2,H,W)
        mag = torch.sqrt(g[:, :1] ** 2 + g[:, 1:2] ** 2 + 1e-12)
        mean = F.conv2d(F.pad(img, (1, 1, 1, 1), mode="replicate"), self.box)
        sq = F.conv2d(F.pad(img ** 2, (1, 1, 1, 1), mode="replicate"), self.box)
        var = (sq - mean ** 2).clamp_min(0.0).sqrt()
        return torch.cat([img, mag, var], 1)                            # (B,3,H,W)


# --------------------------------------------------------------------------- #
#  ConvNeXt-lite building blocks
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for ``(B,C,H,W)`` maps."""

    def __init__(self, c: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + 1e-6)
        return x * self.g[None, :, None, None] + self.b[None, :, None, None]


class ConvNeXtBlock(nn.Module):
    """Depthwise 7x7 + LN + pointwise (C->4C->C) inverted bottleneck, residual."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, 4 * c, 1); self.act = nn.GELU(); self.pw2 = nn.Conv2d(4 * c, c, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class Downsample(nn.Module):
    """LN + stride-2 conv (channel change), halving spatial resolution."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm = LayerNorm2d(cin); self.conv = nn.Conv2d(cin, cout, 2, stride=2)

    def forward(self, x):
        return self.conv(self.norm(x))


# --------------------------------------------------------------------------- #
#  The tokenizer
# --------------------------------------------------------------------------- #
class EdgeFPNTokenizer(nn.Module):
    """Grayscale frame -> learned multi-scale tokens (+ a finest feature map).

    Returns from :meth:`forward`:
      * ``enc_tokens`` ``(B, T, d)`` and ``enc_centres`` ``(B, T, 2)`` — the strided
        FPN levels given to the Perceiver encoder (bounded token count);
      * ``fine_map`` ``(B, d, gf, gf)`` — the stride-4 FPN feature projected to ``d``,
        used by the decoder's windowed neighbour gather (NOT fed to the encoder).
    """

    def __init__(self, d: int = 320, img_size: int = 512,
                 widths=(64, 96, 160, 224), depths=(2, 2, 4, 2),
                 fpn_dim: int = 128, fourier_bands: int = 6,
                 enc_levels=(1, 2, 3), fine_level: int = 0):
        super().__init__()
        self.d, self.img_size, self.fb = d, int(img_size), fourier_bands
        self.top = int(img_size)                       # infer.py reads this for the resize size
        self.widths = tuple(int(w) for w in widths)
        self.fpn_dim = int(fpn_dim)
        self.enc_levels = tuple(int(l) for l in enc_levels)   # which FPN levels feed the encoder
        self.fine_level = int(fine_level)                     # which FPN level feeds the decoder
        self.strides = (4, 8, 16, 32)                         # stride of each FPN level

        self.adapter = EdgeChannels()
        # patchify stem: 3 -> w0, stride 4
        w0 = self.widths[0]
        self.stem = nn.Sequential(nn.Conv2d(3, w0, 4, stride=4), LayerNorm2d(w0))
        # stages (ConvNeXt blocks) with downsamples between them
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, (c, n) in enumerate(zip(self.widths, depths)):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(c) for _ in range(n)]))
            if i < len(self.widths) - 1:
                self.downs.append(Downsample(c, self.widths[i + 1]))
        # FPN: lateral 1x1 to fpn_dim + 3x3 smooth at every level
        self.lat = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in self.widths])
        self.smooth = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in self.widths])
        # shared token projection + per-level embedding
        self.proj = nn.Sequential(nn.Linear(fpn_dim, d), nn.LayerNorm(d))
        self.level_emb = nn.Parameter(torch.zeros(len(self.widths), d))
        self.pos = nn.Sequential(nn.Linear(2 * 2 * fourier_bands, d), nn.LayerNorm(d))

    # -- helpers ----------------------------------------------------------- #
    def _centres(self, g: int, device) -> torch.Tensor:
        ax = (torch.arange(g, device=device) + 0.5) / g * 2 - 1
        cy, cx = torch.meshgrid(ax, ax, indexing="ij")
        return torch.stack([cx, cy], -1).reshape(g * g, 2)              # (g*g, 2)

    def _fpn(self, img):
        """Run the conv encoder + FPN; return a list of ``(B, fpn_dim, g, g)`` per level."""
        x = self.stem(self.adapter(img))
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        # top-down FPN
        lats = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + F.interpolate(lats[i + 1], size=lats[i].shape[-2:], mode="nearest")
        return [s(l) for s, l in zip(self.smooth, lats)]               # smoothed per level

    def _tokens(self, fmap):
        """``(B, fpn_dim, g, g)`` -> tokens ``(B, g*g, d)`` (+ centres) with level/pos tags."""
        B, _, g, _ = fmap.shape
        t = fmap.flatten(2).transpose(1, 2)                            # (B, g*g, fpn_dim)
        cen = self._centres(g, fmap.device)[None].expand(B, -1, -1)
        tok = self.proj(t) + self.pos(fourier_encode(cen, self.fb))
        return tok, cen

    # -- forward ----------------------------------------------------------- #
    def forward(self, img):
        if img.dim() == 3:
            img = img[:, None]
        if img.shape[-2:] != (self.img_size, self.img_size):           # square so the FPN/decoder grids align
            mode = "area" if min(img.shape[-2:]) >= self.img_size else "bilinear"
            kw = {} if mode == "area" else {"align_corners": False}
            img = F.interpolate(img, size=(self.img_size, self.img_size), mode=mode, **kw)
        pyr = self._fpn(img)                                           # 4 levels (strides 4..32)
        enc_toks, enc_cens = [], []
        for li in self.enc_levels:
            tok, cen = self._tokens(pyr[li])
            enc_toks.append(tok + self.level_emb[li]); enc_cens.append(cen)
        enc_tokens = torch.cat(enc_toks, 1)
        enc_centres = torch.cat(enc_cens, 1)
        fine = pyr[self.fine_level]
        B = fine.shape[0]
        fine_map = (self.proj(fine.flatten(2).transpose(1, 2)) + self.level_emb[self.fine_level]
                    ).transpose(1, 2).reshape(B, self.d, fine.shape[-2], fine.shape[-1])
        return enc_tokens, enc_centres, fine_map

    def n_tokens(self) -> int:
        g0 = self.img_size // self.strides[0]
        return sum((g0 * self.strides[0] // self.strides[li]) ** 2 for li in self.enc_levels)


def windowed_neighbors(fine_map: torch.Tensor, s: int, w: int) -> torch.Tensor:
    """Gather each of an ``s×s`` query lattice's ``w×w`` token neighbours from a regular
    feature map ``(B, C, gf, gf)`` (``gf`` divisible by ``s``) -> ``(B, s*s, w*w, C)``.

    Because both the FPN tokens and the decoder query lattice live on aligned regular
    grids, neighbour selection is a fixed ``unfold`` window — exact, and far cheaper than
    an ``O(Q·T)`` ``cdist`` over the 16 k finest tokens.
    """
    B, C, gf, _ = fine_map.shape
    assert gf % s == 0, f"fine grid {gf} must be divisible by query lattice {s}"
    cols = F.unfold(fine_map, kernel_size=w, padding=w // 2)           # (B, C*w*w, gf*gf)
    cols = cols.view(B, C, w * w, gf, gf)
    st = gf // s
    cols = cols[:, :, :, st // 2::st, st // 2::st]                     # (B, C, w*w, s, s)
    return cols.permute(0, 3, 4, 2, 1).reshape(B, s * s, w * w, C)     # (B, s*s, w*w, C)
