"""WaveletSpaceNet — sparse context + a wavelet image pyramid -> mesh-plane + camera pose.

This is the *Points-as-(Super)Tori* idea carried from surfaces to **scenes**.  The
surface model ``PerceiverWaveNet`` reads a ``[context | SEP | main]`` token sequence
(an FPS summary of the whole shape, a learned separator, the dense region) and a
position-conditioned decoder *emits the Haar coefficients* of a distance field.
WaveletSpaceNet keeps that skeleton but swaps the modalities:

    sequence : [ sparse-context points | SEP | wavelet image-pyramid tokens ]
    encoder  : M latents cross-attend the sequence, then L self-attention blocks
    heads    : (1) camera pose  -> 6-D rotation + translation, RELATIVE to the context
               (2) mesh-plane   -> per-pixel depth emitted as 2-D Haar coefficients,
                                   inverted to a depth map and unprojected to a grid mesh

Both heads start at a sane identity: zero-init coefficient heads give a flat plane at
the mean scene depth, and a bias-initialised pose head starts at ``R = I, t = 0`` (the
source viewpoint).  Context may be empty — then the sequence is just ``[SEP | image]``
and the pose is predicted in the frame the network anchors to.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .blocks import CrossBlock, SelfBlock, LocalAttn, fourier_encode, fps
from .wavelet2d import WaveletPyramidTokenizer, idwt2d, haar_filters_2d, DEFAULT_LEVELS
from .geometry import rot6d_to_matrix

# DIODE log-depth stats (mean, std) — see data/diode/log_stats.json in the precursor repo.
DIODE_LOG_STATS = (1.7927592992782593, 1.0100667476654053)


class WaveletSpaceNet(nn.Module):
    def __init__(self, d: int = 256, M: int = 256, L: int = 6, heads: int = 8,
                 levels=DEFAULT_LEVELS, plane_res: int = 64, k: int = 12,
                 n_ctx: int = 48, fourier_bands: int = 8, pos_scale: float = 5.0,
                 log_stats=DIODE_LOG_STATS, pool: int = 4):
        super().__init__()
        assert plane_res % 2 == 0, "plane_res must be even (one DWT halving)"
        self.d, self.k, self.plane_res, self.n_ctx = d, k, plane_res, n_ctx
        self.M, self.L, self.heads = M, L, heads
        self.fb, self.pos_scale = fourier_bands, pos_scale
        self.r = plane_res // 2

        # ---- tokenizers -----------------------------------------------------
        self.img_tok = WaveletPyramidTokenizer(d=d, levels=levels, pool=pool, fourier_bands=6)
        fdim = 3 * 2 * fourier_bands
        self.ctx_tok = nn.Sequential(nn.Linear(fdim, d), nn.LayerNorm(d))
        self.sep = nn.Parameter(torch.randn(1, d) * 0.02)
        self.type_emb = nn.Parameter(torch.zeros(2, d))           # 0 = context, 1 = image

        # ---- Perceiver encoder ---------------------------------------------
        self.latents = nn.Parameter(torch.randn(M, d) * 0.02)
        self.enc_in = CrossBlock(d, heads)
        self.enc = nn.ModuleList([SelfBlock(d, heads) for _ in range(L)])

        # ---- pose head ------------------------------------------------------
        self.pose_q = nn.Parameter(torch.randn(1, d) * 0.02)
        self.pose_x = CrossBlock(d, heads)
        self.pose_head = nn.Linear(d, 9)                          # [tx,ty,tz, rot6d(6)]
        nn.init.zeros_(self.pose_head.weight)
        with torch.no_grad():
            self.pose_head.bias.copy_(torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32))

        # ---- mesh-plane decoder (position-conditioned, wavelet-emitting) ----
        qdim = 2 * 2 * fourier_bands
        self.qemb = nn.Sequential(nn.Linear(qdim, d), nn.LayerNorm(d))
        self.coarse_x = CrossBlock(d, heads)                      # query <- global latents (LL band)
        self.detail_x = LocalAttn(d, heads)                       # query <- nearest image tokens (LH/HL/HH)
        self.coarse_head = nn.Linear(d, 1)
        self.detail_head = nn.Linear(d, 3)
        for h in (self.coarse_head, self.detail_head):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)      # identity start -> flat mean-depth plane

        self.register_buffer("haar2d", haar_filters_2d())
        self.register_buffer("log_mean", torch.tensor(float(log_stats[0])))
        self.register_buffer("log_std", torch.tensor(float(log_stats[1])))
        # the r x r mesh-plane query lattice in [-1, 1] image-plane coords (not learned)
        ax = (torch.arange(self.r) + 0.5) / self.r * 2 - 1
        qy, qx = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("qpos", torch.stack([qx, qy], -1).reshape(-1, 2))   # (r*r, 2)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def depth_from_lognorm(self, logn: torch.Tensor) -> torch.Tensor:
        return torch.exp(logn * self.log_std + self.log_mean)

    def encode(self, img, ctx_P=None, n_ctx=None):
        """Build the token sequence and run the Perceiver encoder -> latents, image tokens, centres."""
        B = img.shape[0]
        img_t, img_cen = self.img_tok(img)                       # (B,T,d), (B,T,2)
        img_t = img_t + self.type_emb[1]
        toks = [self.sep[None].expand(B, -1, -1), img_t]
        nctx = self.n_ctx if n_ctx is None else int(n_ctx)
        if ctx_P is not None and ctx_P.shape[1] > 0 and nctx > 0:
            idx = fps(ctx_P, min(nctx, ctx_P.shape[1]))
            Pg = torch.gather(ctx_P, 1, idx[..., None].expand(-1, -1, 3))
            ct = self.ctx_tok(fourier_encode(Pg / self.pos_scale, self.fb)) + self.type_emb[0]
            toks = [ct] + toks
        seq = torch.cat(toks, 1)
        lat = self.enc_in(self.latents[None].expand(B, -1, -1), seq)
        for blk in self.enc:
            lat = blk(lat)
        return lat, img_t, img_cen

    def forward(self, img, ctx_P=None, n_ctx=None):
        if img.dim() == 3:
            img = img[:, None]
        B, dev = img.shape[0], img.device
        lat, img_t, img_cen = self.encode(img, ctx_P, n_ctx)

        # ---- pose ----
        pq = self.pose_x(self.pose_q[None].expand(B, -1, -1), lat)[:, 0]    # (B,d)
        pose = self.pose_head(pq)
        t = pose[:, :3] * self.pos_scale
        R = rot6d_to_matrix(pose[:, 3:9])

        # ---- mesh-plane (position-conditioned wavelet decode) ----
        q = self.qemb(fourier_encode(self.qpos / 1.0, self.fb))[None].expand(B, -1, -1)   # (B,Q,d)
        cf = self.coarse_x(q, lat)                                 # global -> coarse
        kk = min(self.k, img_t.shape[1])
        d2 = torch.cdist(self.qpos[None].expand(B, -1, -1), img_cen)        # (B,Q,T)
        tk = d2.topk(kk, dim=-1, largest=False)
        nbr = torch.gather(img_t, 1, tk.indices.reshape(B, -1, 1).expand(-1, -1, self.d)
                           ).reshape(B, self.qpos.shape[0], kk, self.d)
        df = self.detail_x(q, nbr)                                 # local -> detail
        c_ll = self.coarse_head(cf)                                # (B,Q,1)
        c_det = self.detail_head(df)                               # (B,Q,3)
        coeffs = torch.cat([c_ll, c_det], -1).view(B, self.r, self.r, 4).permute(0, 3, 1, 2)
        logn = idwt2d(coeffs, self.haar2d)                         # (B,1,plane_res,plane_res)
        depth = self.depth_from_lognorm(logn)
        return {"R": R, "t": t, "rot6d": pose[:, 3:9], "logdepth": logn,
                "depth": depth, "coeffs": coeffs}


def save_checkpoint(net: WaveletSpaceNet, path: str, **meta):
    cfg = dict(d=net.d, M=net.M, L=net.L, heads=net.heads, plane_res=net.plane_res,
               k=net.k, n_ctx=net.n_ctx, fourier_bands=net.fb, pos_scale=net.pos_scale,
               levels=net.img_tok.levels, pool=net.img_tok.pool,
               log_stats=(float(net.log_mean), float(net.log_std)))
    torch.save({"state": net.state_dict(), "cfg": cfg, **meta}, path)


def load_checkpoint(path, map_location="cpu", **override):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ck["cfg"]); cfg.update(override)
    net = WaveletSpaceNet(**cfg)
    net.load_state_dict(ck["state"])
    return net, ck
