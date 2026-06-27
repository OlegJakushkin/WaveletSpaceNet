"""Losses + metrics for WaveletSpaceNet v2 — rewritten to *reward* small surfaces.

v1's terms were all global masked means, so flat walls/floor set the gradient and a chair
contributed a negligible fraction; the flat-plane init was a near-optimum.  v2 adds the
incentives that small/thin structures need:

* **edge-weighted depth L1** — per-pixel weight ``1 + alpha*|grad(log-depth)|`` so object
  boundaries (where thin objects live) count far more than flat interiors;
* **multi-scale gradient matching** — directly supervises the sharp depth steps that *are*
  small objects (the Eigen/MegaDepth term);
* **multi-level wavelet** — matches the decoder's emitted Haar pyramid, up-weighting the
  detail bands and densifying the holey splat GT before analysis;
* **bidirectional chamfer** for eval/selection, so *failing to cover* a thin object is
  finally penalised (v1's one-directional metric rewarded smoothing furniture away).

The camera-pose terms (translation L1 + rotation geodesic) are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .wavelet2d import haar_analysis
from . import geometry as G


def _lognorm(depth, log_mean, log_std, eps=1e-3):
    return (torch.log(depth.clamp_min(eps)) - log_mean) / log_std


def _fill_holes(x, m, iters=16):
    """Propagate valid values into masked holes (iterative masked blur) so the wavelet
    target isn't dominated by hard zero-edges at the splat's holes."""
    filled = x * m
    valid = m.clone()
    k = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0
    for _ in range(iters):
        num = F.conv2d(F.pad(filled, (1, 1, 1, 1), mode="replicate"), k)
        den = F.conv2d(F.pad(valid, (1, 1, 1, 1), mode="replicate"), k)
        newv = (den > 0).float()
        filled = torch.where(valid > 0.5, filled, num / den.clamp_min(1e-6) * newv)
        valid = torch.clamp(valid + newv, max=1.0)
    return filled


def _edge_weight(gln, mask, alpha=4.0, wmax=6.0):
    """Per-pixel weight ``1 + alpha*|grad(gln)|`` (clamped), zeroed where a finite-diff
    stencil would cross an invalid pixel — large at depth discontinuities / thin objects."""
    dx = (gln[..., :, 1:] - gln[..., :, :-1]).abs()
    dy = (gln[..., 1:, :] - gln[..., :-1, :]).abs()
    mdx = mask[..., :, 1:] * mask[..., :, :-1]
    mdy = mask[..., 1:, :] * mask[..., :-1, :]
    gx = F.pad(dx * mdx, (1, 0, 0, 0)); gy = F.pad(dy * mdy, (0, 0, 1, 0))
    g = torch.maximum(gx, gy)
    return (1.0 + alpha * g).clamp(max=wmax) * mask


def _multiscale_grad(pln, gln, mask, scales=(1, 2, 4)):
    """Mean over scales of masked one-sided gradient-matching L1 (subsampled per scale)."""
    tot = pln.new_zeros(())
    for s in scales:
        p = pln[..., ::s, ::s]; g = gln[..., ::s, ::s]; m = mask[..., ::s, ::s]
        dpx = p[..., :, 1:] - p[..., :, :-1]; dgx = g[..., :, 1:] - g[..., :, :-1]
        dpy = p[..., 1:, :] - p[..., :-1, :]; dgy = g[..., 1:, :] - g[..., :-1, :]
        mx = m[..., :, 1:] * m[..., :, :-1]; my = m[..., 1:, :] * m[..., :-1, :]
        lx = ((dpx - dgx).abs() * mx).sum() / mx.sum().clamp_min(1.0)
        ly = ((dpy - dgy).abs() * my).sum() / my.sum().clamp_min(1.0)
        tot = tot + lx + ly
    return tot / len(scales)


def _pool_mask(mask, side):
    return (F.adaptive_avg_pool2d(mask, side) > 0.5).float()


def _wavelet_loss(pred, gln, mask, net, det_band_w=2.0):
    """Multi-level Haar coefficient L1 against the densified GT, detail bands up-weighted."""
    gln_f = _fill_holes(gln, mask)
    ll0_gt, dets_gt = haar_analysis(gln_f, net.wave_levels, net.haar2d)
    mll = _pool_mask(mask, ll0_gt.shape[-1])
    num = ((pred["ll0"] - ll0_gt).abs() * mll).sum()
    den = mll.sum().clamp_min(1.0)
    for dp, dg in zip(pred["dets"], dets_gt):
        md = _pool_mask(mask, dg.shape[-1])
        num = num + det_band_w * ((dp - dg).abs() * md).sum()
        den = den + det_band_w * md.sum() * dg.shape[1]      # 3 detail channels per cell
    return num / den.clamp_min(1.0)


def _normals_from_depth(depth, K):
    B, _, h, w = depth.shape
    p = G.depth_to_points(depth, K).reshape(B, h, w, 3).permute(0, 3, 1, 2)
    dx = p[:, :, :, 2:] - p[:, :, :, :-2]
    dy = p[:, :, 2:, :] - p[:, :, :-2, :]
    dx = F.pad(dx, (1, 1, 0, 0)); dy = F.pad(dy, (0, 0, 1, 1))
    return F.normalize(torch.cross(dx, dy, dim=1), dim=1, eps=1e-6)


def space_loss(pred, batch, net, *, lam_depth=1.0, lam_grad=1.0, lam_wave=0.5,
               lam_normal=0.2, lam_trans=1.0, lam_rot=1.0, edge_alpha=4.0):
    """Composite training loss.  Returns ``(loss, parts)`` (parts = detached floats)."""
    dev = pred["depth"].device
    gt_depth = batch["depth"].to(dev)                 # (B,1,P,P) metres, 0 = invalid
    mask = (batch["mask"].to(dev) > 0.5).float()
    K = batch["K_plane"].to(dev)
    Rg, tg = batch["R"].to(dev), batch["t"].to(dev)

    gln = _lognorm(gt_depth, net.log_mean, net.log_std)
    pln = pred["logdepth"]

    # ---- edge-weighted log-depth L1 (flat walls stop owning the gradient) ----
    w = _edge_weight(gln, mask, alpha=edge_alpha)
    l_depth = ((pln - gln).abs() * w).sum() / w.sum().clamp_min(1.0)

    # ---- multi-scale gradient matching (rewards the sharp steps that ARE small objects) ----
    l_grad = _multiscale_grad(pln, gln, mask)

    # ---- multi-level wavelet (detail bands up-weighted, GT densified) ----
    l_wave = _wavelet_loss(pred, gln, mask, net)

    # ---- normal-from-depth (low weight, masked) ----
    npred = _normals_from_depth(pred["depth"], K)
    ngt = _normals_from_depth(gt_depth.clamp_min(1e-3), K)
    cos = (npred * ngt).sum(1, keepdim=True)
    l_normal = ((1 - cos) * mask).sum() / mask.sum().clamp_min(1.0)

    # ---- camera pose ----
    l_trans = (pred["t"] - tg).abs().mean()
    l_rot = G.geodesic_angle(pred["R"], Rg).mean()

    loss = (lam_depth * l_depth + lam_grad * l_grad + lam_wave * l_wave
            + lam_normal * l_normal + lam_trans * l_trans + lam_rot * l_rot)
    parts = {"loss": float(loss.detach()), "depth": float(l_depth.detach()),
             "grad": float(l_grad.detach()), "wave": float(l_wave.detach()),
             "normal": float(l_normal.detach()), "trans": float(l_trans.detach()),
             "rot": float(l_rot.detach())}
    return loss, parts


@torch.no_grad()
def chamfer_to_scene(pred, batch, net, max_pred=2048):
    """**Bidirectional** chamfer (m): the symmetric mean of pred->scene and scene->pred
    nearest distances, so failing to COVER a thin object is penalised (not just drift)."""
    dev = pred["depth"].device
    K = batch["K_plane"].to(dev)
    # fp32 for numerically-stable cdist (model outputs may be bf16 under autocast)
    pts = G.depth_to_points(pred["depth"].float(), K, pred["R"].float(), pred["t"].float())  # (B,Q,3) world
    mask = (batch["mask"].to(dev) > 0.5).reshape(pts.shape[0], -1)
    sp = batch["scene_pts"].to(dev); spn = batch.get("scene_n")
    tot, cnt = 0.0, 0
    for b in range(pts.shape[0]):
        valid = pts[b][mask[b]]
        if valid.shape[0] == 0:
            continue
        if valid.shape[0] > max_pred:
            valid = valid[torch.randperm(valid.shape[0], device=dev)[:max_pred]]
        ns = int(spn[b]) if spn is not None else sp.shape[1]
        ref = sp[b, :ns]
        if ref.shape[0] == 0:
            continue
        d = torch.cdist(valid, ref)
        fwd = float(d.min(1).values.mean())            # pred -> scene (accuracy)
        bwd = float(d.min(0).values.mean())            # scene -> pred (coverage / recall)
        tot += 0.5 * (fwd + bwd); cnt += 1
    return tot / max(cnt, 1)


@torch.no_grad()
def eval_metrics(pred, batch, net):
    """Validation metrics: masked log-depth L1 (normalised) and symmetric chamfer(m)."""
    dev = pred["depth"].device
    gt_depth = batch["depth"].to(dev)
    mask = (batch["mask"].to(dev) > 0.5).float()
    gln = _lognorm(gt_depth, net.log_mean, net.log_std)
    logdepth_l1 = float(((pred["logdepth"].float() - gln).abs() * mask).sum() / mask.sum().clamp_min(1.0))
    return {"logdepthL1": logdepth_l1, "chamfer": chamfer_to_scene(pred, batch, net)}
