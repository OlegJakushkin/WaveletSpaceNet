"""Losses + metrics for WaveletSpaceNet.

The reconstruction terms mirror the precursor monocular-scene model (log-depth L1 +
normal-from-depth + wavelet-coefficient), and two terms are added for this task:

* **camera pose** — translation L1 + rotation geodesic angle, supervising the pose head
  relative to the context frame;
* **chamfer (m)** — the eval metric: how far the unprojected mesh-plane lands from the
  real scene cloud, in metres.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .wavelet2d import dwt2d
from . import geometry as G


def _normals_from_depth(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """``depth (B,1,h,w)`` -> per-pixel unit normals ``(B,3,h,w)`` in the camera frame."""
    B, _, h, w = depth.shape
    p = G.depth_to_points(depth, K).reshape(B, h, w, 3).permute(0, 3, 1, 2)   # (B,3,h,w)
    dx = p[:, :, :, 2:] - p[:, :, :, :-2]
    dy = p[:, :, 2:, :] - p[:, :, :-2, :]
    dx = F.pad(dx, (1, 1, 0, 0)); dy = F.pad(dy, (0, 0, 1, 1))
    n = torch.cross(dx, dy, dim=1)
    return F.normalize(n, dim=1, eps=1e-6)


def _lognorm(depth, log_mean, log_std, eps=1e-3):
    return (torch.log(depth.clamp_min(eps)) - log_mean) / log_std


def space_loss(pred, batch, net, *, lam_depth=1.0, lam_normal=0.5, lam_wave=0.5,
               lam_trans=1.0, lam_rot=1.0):
    """Composite training loss.  Returns ``(loss, parts)`` (parts = detached floats)."""
    dev = pred["depth"].device
    gt_depth = batch["depth"].to(dev)                 # (B,1,r,r) metres, 0 = invalid
    mask = (batch["mask"].to(dev) > 0.5).float()
    K = batch["K_plane"].to(dev)
    Rg, tg = batch["R"].to(dev), batch["t"].to(dev)
    m_sum = mask.sum().clamp_min(1.0)

    # ---- log-depth L1 (masked) ----
    gln = _lognorm(gt_depth, net.log_mean, net.log_std)
    pln = pred["logdepth"]
    l_depth = ((pln - gln).abs() * mask).sum() / m_sum

    # ---- normal-from-depth (masked cosine) ----
    npred = _normals_from_depth(pred["depth"], K)
    ngt = _normals_from_depth(gt_depth.clamp_min(1e-3), K)
    cos = (npred * ngt).sum(1, keepdim=True)
    l_normal = ((1 - cos) * mask).sum() / m_sum

    # ---- wavelet-coefficient L1 (fill invalid GT with the prediction so holes cost ~0) ----
    gln_filled = torch.where(mask > 0.5, gln, pln.detach())
    tgt_c = dwt2d(gln_filled, net.haar2d)
    mask_c = F.avg_pool2d(mask, 2)                    # band-level validity weight
    l_wave = ((pred["coeffs"] - tgt_c).abs() * mask_c).sum() / mask_c.sum().clamp_min(1.0)

    # ---- camera pose ----
    l_trans = (pred["t"] - tg).abs().mean()
    l_rot = G.geodesic_angle(pred["R"], Rg).mean()

    loss = (lam_depth * l_depth + lam_normal * l_normal + lam_wave * l_wave
            + lam_trans * l_trans + lam_rot * l_rot)
    parts = {"loss": float(loss.detach()), "depth": float(l_depth.detach()),
             "normal": float(l_normal.detach()), "wave": float(l_wave.detach()),
             "trans": float(l_trans.detach()), "rot": float(l_rot.detach())}
    return loss, parts


@torch.no_grad()
def chamfer_to_scene(pred, batch, net, max_pred=2048):
    """One-directional chamfer (m): mean distance from masked mesh-plane points (placed in
    the context frame by the *predicted* pose) to the real scene cloud."""
    dev = pred["depth"].device
    K = batch["K_plane"].to(dev)
    pts = G.depth_to_points(pred["depth"], K, pred["R"], pred["t"])     # (B,Q,3) world
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
        d = torch.cdist(valid, ref).min(1).values
        tot += float(d.mean()); cnt += 1
    return tot / max(cnt, 1)


@torch.no_grad()
def eval_metrics(pred, batch, net):
    """Validation metrics: masked log-depth L1 (normalised) and chamfer(m)."""
    dev = pred["depth"].device
    gt_depth = batch["depth"].to(dev)
    mask = (batch["mask"].to(dev) > 0.5).float()
    gln = _lognorm(gt_depth, net.log_mean, net.log_std)
    logdepth_l1 = float(((pred["logdepth"] - gln).abs() * mask).sum() / mask.sum().clamp_min(1.0))
    return {"logdepthL1": logdepth_l1, "chamfer": chamfer_to_scene(pred, batch, net)}
