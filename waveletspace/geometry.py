"""Camera geometry: pinhole intrinsics, poses, (un)projection, splat rendering and
smooth-spline fly-throughs.

Convention (OpenCV): camera axes are **x right, y down, z forward** (into the scene),
so a visible point has ``z > 0`` and projects to ``u = fx*x/z + cx``, ``v = fy*y/z + cy``.
A pose is the camera-to-world rigid transform ``(R, t)``: ``world = R @ p_cam + t`` and
``p_cam = R^T @ (world - t)``.  Because the data pipeline unprojects and re-projects with
the *same* intrinsics, absolute focal length is a free, self-consistent choice.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

WORLD_UP = np.array([0.0, -1.0, 0.0], np.float64)        # y is down in image space -> world "up" is -y


# --------------------------------------------------------------------------- #
#  Intrinsics
# --------------------------------------------------------------------------- #
def intrinsics(H: int, W: int, vfov_deg: float = 60.0) -> np.ndarray:
    """Pinhole ``K`` (3x3) for an ``H×W`` image with the given vertical field of view."""
    fy = 0.5 * H / np.tan(np.radians(vfov_deg) * 0.5)
    fx = fy                                              # square pixels
    return np.array([[fx, 0, (W - 1) * 0.5], [0, fy, (H - 1) * 0.5], [0, 0, 1]], np.float64)


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Rescale ``K`` for an image resized by ``(sx, sy)``."""
    K = K.copy(); K[0] *= sx; K[1] *= sy
    return K


# --------------------------------------------------------------------------- #
#  Rotations
# --------------------------------------------------------------------------- #
def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = WORLD_UP) -> np.ndarray:
    """Camera-to-world rotation (3x3) looking from ``eye`` toward ``target`` (x-right,y-down,z-fwd)."""
    z = target - eye
    nz = np.linalg.norm(z)
    z = z / nz if nz > 1e-9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    nx = np.linalg.norm(x)
    x = x / nx if nx > 1e-9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)                  # columns = camera axes in world


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. continuous 6-D rotation -> ``(..., 3, 3)`` via Gram-Schmidt."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)            # columns -> rotation matrix


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> 6-D representation (first two columns)."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def geodesic_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between two batched rotation matrices ``(..., 3, 3)``."""
    rel = Ra.transpose(-1, -2) @ Rb
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return torch.arccos(torch.clamp((tr - 1) * 0.5, -1 + 1e-6, 1 - 1e-6))


# --------------------------------------------------------------------------- #
#  Projection / unprojection
# --------------------------------------------------------------------------- #
def unproject_depth(depth: np.ndarray, K: np.ndarray, R=None, t=None, mask=None):
    """Depth map ``(H, W)`` -> world points ``(M, 3)`` (+ valid pixel index ``(M, 2)`` as v,u).

    ``R, t`` give the camera-to-world pose (default identity = camera frame).  Pixels with
    non-positive / non-finite depth (or ``mask == 0``) are dropped.
    """
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask > 0.5
    u = uu[valid].astype(np.float64); v = vv[valid].astype(np.float64); z = depth[valid].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * z; y = (v - cy) / fy * z
    p = np.stack([x, y, z], 1)
    if R is not None:
        p = p @ R.T + (t if t is not None else 0.0)
    return p, np.stack([vv[valid], uu[valid]], 1)


def project_points(P: np.ndarray, K: np.ndarray, R=None, t=None):
    """World points ``(M, 3)`` -> pixel ``(u, v)`` floats and camera-frame depth ``z``."""
    if R is not None:
        P = (P - (t if t is not None else 0.0)) @ R           # R^T applied: (P-t) @ R == R^T @ (P-t)
    z = P[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * P[:, 0] / np.maximum(z, 1e-9) + cx
    v = fy * P[:, 1] / np.maximum(z, 1e-9) + cy
    return u, v, z


# --------------------------------------------------------------------------- #
#  Splat renderer (painter's z-buffer)
# --------------------------------------------------------------------------- #
def splat_render(P, gray, K, R, t, H, W, radius: int = 1):
    """Render world points ``P (M,3)`` with grayscale ``gray (M,)`` from pose ``(R, t)``.

    Painter's algorithm: candidates are drawn far-to-near into a small ``(2*radius+1)``
    splat so the nearest surface wins each pixel and sparse clouds leave fewer holes.
    Returns ``(img (H,W) float32, depth (H,W) float32 [0 = empty], mask (H,W) float32)``.
    """
    u, v, z = project_points(P, K, R, t)
    img = np.zeros((H, W), np.float32)
    depth = np.zeros((H, W), np.float32)
    mask = np.zeros((H, W), np.float32)
    front = z > 1e-6
    if not front.any():
        return img, depth, mask
    u, v, z, g = u[front], v[front], z[front], gray[front]
    order = np.argsort(-z)                                    # far first -> near overwrites
    u, v, z, g = u[order], v[order], z[order], g[order]
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu = ui + dx; vv = vi + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[ok], uu[ok]] = g[ok]
            depth[vv[ok], uu[ok]] = z[ok]
            mask[vv[ok], uu[ok]] = 1.0
    return img, depth, mask


# --------------------------------------------------------------------------- #
#  Smooth-spline fly-throughs
# --------------------------------------------------------------------------- #
def _catmull_rom(ctrl: np.ndarray, n: int) -> np.ndarray:
    """Sample ``n`` points along a Catmull-Rom spline through control points ``ctrl (K,3)``."""
    K = len(ctrl)
    pad = np.concatenate([ctrl[:1], ctrl, ctrl[-1:]], 0)     # clamp endpoints
    segs = K - 1
    out = []
    for i in range(n):
        s = i / max(n - 1, 1) * segs
        j = min(int(s), segs - 1)
        u = s - j
        p0, p1, p2, p3 = pad[j], pad[j + 1], pad[j + 2], pad[j + 3]
        u2, u3 = u * u, u * u * u
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    return np.stack(out, 0)


def flythrough(center, extent, rng, n_frames: int = 16, n_ctrl: int = 4,
               eye_anchor=None, move_frac: float = 0.25, jitter: float = 0.05):
    """A randomised smooth fly-through that keeps ``center`` in frame.

    Control eyes are random offsets (scaled by ``extent * move_frac``) around
    ``eye_anchor`` (default the origin = the source viewpoint); a Catmull-Rom spline
    smooths them into ``n_frames`` eye positions; each camera looks at ``center`` with a
    small per-frame jitter.  Returns ``(R (n,3,3), t (n,3))`` camera-to-world poses.
    """
    center = np.asarray(center, np.float64)
    eye_anchor = np.zeros(3) if eye_anchor is None else np.asarray(eye_anchor, np.float64)
    step = float(extent) * move_frac
    ctrl = eye_anchor[None] + rng.normal(0, step, size=(n_ctrl, 3))
    ctrl[0] = eye_anchor                                      # start at the anchor
    eyes = _catmull_rom(ctrl, n_frames)
    tgt_jit = rng.normal(0, float(extent) * jitter, size=(n_frames, 3))
    Rs, ts = [], []
    for i in range(n_frames):
        R = look_at(eyes[i], center + tgt_jit[i])
        Rs.append(R); ts.append(eyes[i])
    return np.stack(Rs, 0), np.stack(ts, 0)


# --------------------------------------------------------------------------- #
#  Mesh-plane from a depth map (torch, batched) — the model's geometric output
# --------------------------------------------------------------------------- #
def depth_to_points(depth: torch.Tensor, K: torch.Tensor, R=None, t=None) -> torch.Tensor:
    """Batched unprojection: ``depth (B,1,h,w)`` -> world points ``(B, h*w, 3)``.

    ``K`` is ``(B,3,3)`` for the ``h×w`` lattice; ``R (B,3,3)``, ``t (B,3)`` the
    camera-to-world pose (default identity).  Differentiable w.r.t. depth and pose.
    """
    B, _, h, w = depth.shape
    dev, dt = depth.device, depth.dtype
    vv, uu = torch.meshgrid(torch.arange(h, device=dev, dtype=dt),
                            torch.arange(w, device=dev, dtype=dt), indexing="ij")
    u = uu.reshape(1, -1).expand(B, -1); v = vv.reshape(1, -1).expand(B, -1)
    z = depth.reshape(B, -1)
    fx = K[:, 0, 0:1]; fy = K[:, 1, 1:2]; cx = K[:, 0, 2:3]; cy = K[:, 1, 2:3]
    x = (u - cx) / fx * z; y = (v - cy) / fy * z
    p = torch.stack([x, y, z], -1)                            # (B, h*w, 3) camera frame
    if R is not None:
        p = torch.einsum("bij,bnj->bni", R, p)
        if t is not None:
            p = p + t[:, None]
    return p


def grid_faces(h: int, w: int) -> np.ndarray:
    """Triangle faces ``(2*(h-1)*(w-1), 3)`` for an ``h×w`` vertex grid (row-major)."""
    idx = np.arange(h * w).reshape(h, w)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel(); d = idx[1:, 1:].ravel()
    return np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)], 0)
