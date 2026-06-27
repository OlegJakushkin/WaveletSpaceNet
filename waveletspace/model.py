"""WaveletSpaceNet v2 ("Sharp-FPN") — learned image embedding + sparse context
-> mesh-plane + camera pose, with small surfaces (tables, chairs, thin objects) as the
explicit design target.

This carries the *Points-as-(Super)Tori* skeleton (a ``[context | SEP | image]`` token
sequence read by Perceiver latents; a position-conditioned decoder that *emits wavelet
coefficients* of the geometry) but fixes the four places where small-surface information
was being destroyed in v1:

    ENCODE    a learned conv-FPN tokenizer (waveletspace.encoder) replaces the Haar-mean
              tokenizer that averaged signed detail bands to ~zero; grayscale + computed
              edge channels keep it sensor-agnostic.
    ROUTE     the coarse (LL) depth band now reads LOCAL image evidence (``ll_local``),
              not only the global latents, so a chair's coarse depth can come from the
              chair's own pixels instead of the scene average.
    REPRESENT the decoder emits a MULTI-LEVEL Haar pyramid (LL0 + J detail octaves) at
              ``plane_res`` 256, so thin objects get coefficients at several scales.
    DETAIL    the detail path is deep+wide (LocalAttn -> conv mix -> LocalAttn over a
              ``win×win`` window of the finest feature map), not one shallow readout.

Identity start is preserved: every coefficient head is zero-initialised (flat plane at the
mean scene depth) and the pose head is bias-initialised to ``R = I, t = 0``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CrossBlock, SelfBlock, LocalAttn, fourier_encode, fps
from .encoder import EdgeFPNTokenizer, LayerNorm2d, windowed_neighbors
from .wavelet2d import haar_filters_2d, haar_synthesis
from .geometry import rot6d_to_matrix

# DIODE log-depth stats (mean, std) — see data/diode/log_stats.json in the precursor repo.
DIODE_LOG_STATS = (1.7927592992782593, 1.0100667476654053)


class ConvRefine(nn.Module):
    """Cheap spatial mixer over the query lattice so neighbouring depth queries share
    edge evidence (a depthwise-3x3 + pointwise inverted bottleneck, residual)."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = LayerNorm2d(d)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.pw1 = nn.Conv2d(d, 2 * d, 1); self.act = nn.GELU(); self.pw2 = nn.Conv2d(2 * d, d, 1)

    def forward(self, x, s):                                   # x:(B,s*s,d)
        B, Q, d = x.shape
        h = x.transpose(1, 2).reshape(B, d, s, s)
        h = self.pw2(self.act(self.pw1(self.dw(self.norm(h)))))
        return x + h.reshape(B, d, s * s).transpose(1, 2)


class WaveletSpaceNet(nn.Module):
    def __init__(self, d: int = 320, M: int = 384, L: int = 8, heads: int = 8,
                 plane_res: int = 256, wave_levels: int = 3, win: int = 5,
                 n_ctx: int = 64, fourier_bands: int = 8, pos_scale: float = 5.0,
                 log_stats=DIODE_LOG_STATS, img_size: int = 512,
                 enc_widths=(64, 96, 160, 224), enc_depths=(2, 2, 4, 2), fpn_dim: int = 128):
        super().__init__()
        self.d, self.M, self.L, self.heads = d, M, L, heads
        self.plane_res, self.wave_levels, self.win = plane_res, wave_levels, win
        self.n_ctx, self.fb, self.pos_scale = n_ctx, fourier_bands, pos_scale
        self.img_size = img_size
        self.enc_widths, self.enc_depths, self.fpn_dim = tuple(enc_widths), tuple(enc_depths), fpn_dim
        self.g0 = plane_res // (2 ** wave_levels)              # coarsest LL side
        assert self.g0 >= 1 and self.g0 * (2 ** wave_levels) == plane_res, \
            "plane_res must be (g0)*2**wave_levels with g0>=1"
        self.sides = [self.g0 * (2 ** j) for j in range(wave_levels)]   # detail lattice sides

        # ---- tokenizers -----------------------------------------------------
        self.img_tok = EdgeFPNTokenizer(d=d, img_size=img_size, widths=enc_widths,
                                        depths=enc_depths, fpn_dim=fpn_dim)
        fdim = 3 * 2 * fourier_bands
        self.ctx_tok = nn.Sequential(nn.Linear(fdim, d), nn.LayerNorm(d))
        self.sep = nn.Parameter(torch.randn(1, d) * 0.02)
        self.type_emb = nn.Parameter(torch.zeros(2, d))       # 0 = context, 1 = image

        # ---- Perceiver encoder ---------------------------------------------
        self.latents = nn.Parameter(torch.randn(M, d) * 0.02)
        self.enc_in = CrossBlock(d, heads)
        self.enc = nn.ModuleList([SelfBlock(d, heads) for _ in range(L)])

        # ---- pose head ------------------------------------------------------
        self.pose_q = nn.Parameter(torch.randn(1, d) * 0.02)
        self.pose_x = CrossBlock(d, heads)
        self.pose_head = nn.Linear(d, 9)                      # [tx,ty,tz, rot6d(6)]
        nn.init.zeros_(self.pose_head.weight)
        with torch.no_grad():
            self.pose_head.bias.copy_(torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32))

        # ---- mesh-plane decoder (multi-level, wavelet-emitting) -------------
        qdim = 2 * 2 * fourier_bands
        self.qemb = nn.Sequential(nn.Linear(qdim, d), nn.LayerNorm(d))
        self.coarse_x = CrossBlock(d, heads)                  # coarse query <- global latents
        self.local1 = LocalAttn(d, heads)                     # detail: query <- finest tokens
        self.refine = ConvRefine(d)                           # neighbouring queries share evidence
        self.local2 = LocalAttn(d, heads)
        self.coarse_ll_head = nn.Linear(d, 1)                 # LL0 from global
        self.ll_local_head = nn.Linear(d, 1)                  # + LL0 from LOCAL evidence (the routing fix)
        self.coarse_det_head = nn.Linear(d, 3)                # coarsest detail octave (global)
        self.det_local_head = nn.Linear(d, 3)                 # finer detail octaves (local)
        for h in (self.coarse_ll_head, self.ll_local_head, self.coarse_det_head, self.det_local_head):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)  # identity start -> flat mean-depth plane

        self.register_buffer("haar2d", haar_filters_2d())
        self.register_buffer("log_mean", torch.tensor(float(log_stats[0])))
        self.register_buffer("log_std", torch.tensor(float(log_stats[1])))
        for j, s in enumerate(self.sides):                    # per-level query lattices in [-1,1]
            ax = (torch.arange(s) + 0.5) / s * 2 - 1
            qy, qx = torch.meshgrid(ax, ax, indexing="ij")
            self.register_buffer(f"qpos{j}", torch.stack([qx, qy], -1).reshape(-1, 2))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def depth_from_lognorm(self, logn: torch.Tensor) -> torch.Tensor:
        return torch.exp(logn * self.log_std + self.log_mean)

    def encode(self, img, ctx_P=None, n_ctx=None):
        """Build the token sequence, run the Perceiver encoder -> (latents, finest feature map)."""
        B = img.shape[0]
        enc_tok, _enc_cen, fine_map = self.img_tok(img)       # (B,T,d), (B,T,2), (B,d,gf,gf)
        enc_tok = enc_tok + self.type_emb[1]
        toks = [self.sep[None].expand(B, -1, -1), enc_tok]
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
        return lat, fine_map

    def _local(self, qpos, fine_map, s):
        """Deep local readout for an ``s×s`` query lattice from the finest feature map."""
        B = fine_map.shape[0]
        q = self.qemb(fourier_encode(qpos / 1.0, self.fb))[None].expand(B, -1, -1)   # (B,s*s,d)
        nbr = windowed_neighbors(fine_map, s, self.win)       # (B,s*s,win*win,d)
        h = self.local1(q, nbr)
        h = self.refine(h, s)
        return self.local2(h, nbr)                            # (B,s*s,d)

    def forward(self, img, ctx_P=None, n_ctx=None):
        if img.dim() == 3:
            img = img[:, None]
        B = img.shape[0]
        lat, fine_map = self.encode(img, ctx_P, n_ctx)

        # ---- pose ----
        pq = self.pose_x(self.pose_q[None].expand(B, -1, -1), lat)[:, 0]     # (B,d)
        pose = self.pose_head(pq)
        t = pose[:, :3] * self.pos_scale
        R = rot6d_to_matrix(pose[:, 3:9])

        # ---- mesh-plane: coarse octave (global + local) then finer detail octaves ----
        g0 = self.g0
        q0 = self.qemb(fourier_encode(getattr(self, "qpos0") / 1.0, self.fb))[None].expand(B, -1, -1)
        cf0 = self.coarse_x(q0, lat)                          # global -> coarse
        l0 = self._local(getattr(self, "qpos0"), fine_map, g0)              # local at g0
        ll0 = self.coarse_ll_head(cf0) + self.ll_local_head(l0)            # (B,g0*g0,1)
        dets = [self.coarse_det_head(cf0)]                                  # coarsest detail (B,g0*g0,3)
        for j in range(1, self.wave_levels):
            s = self.sides[j]
            lj = self._local(getattr(self, f"qpos{j}"), fine_map, s)
            dets.append(self.det_local_head(lj))             # (B,s*s,3)

        ll0_map = ll0.view(B, g0, g0, 1).permute(0, 3, 1, 2)
        det_maps = [d.view(B, s, s, 3).permute(0, 3, 1, 2) for d, s in zip(dets, self.sides)]
        logn = haar_synthesis(ll0_map, det_maps, self.haar2d)              # (B,1,plane_res,plane_res)
        depth = self.depth_from_lognorm(logn)
        return {"R": R, "t": t, "rot6d": pose[:, 3:9], "logdepth": logn,
                "depth": depth, "ll0": ll0_map, "dets": det_maps}


def save_checkpoint(net: WaveletSpaceNet, path: str, **meta):
    cfg = dict(d=net.d, M=net.M, L=net.L, heads=net.heads, plane_res=net.plane_res,
               wave_levels=net.wave_levels, win=net.win, n_ctx=net.n_ctx,
               fourier_bands=net.fb, pos_scale=net.pos_scale,
               img_size=net.img_size, enc_widths=net.enc_widths, enc_depths=net.enc_depths,
               fpn_dim=net.fpn_dim, log_stats=(float(net.log_mean), float(net.log_std)))
    torch.save({"state": net.state_dict(), "cfg": cfg, **meta}, path)


def load_checkpoint(path, map_location="cpu", **override):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ck["cfg"]); cfg.update(override)
    cfg.pop("levels", None); cfg.pop("pool", None); cfg.pop("k", None)   # drop v1-only fields
    net = WaveletSpaceNet(**cfg)
    net.load_state_dict(ck["state"])
    return net, ck
