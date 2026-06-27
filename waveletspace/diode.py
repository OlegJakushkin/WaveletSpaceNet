"""DIODE fly-through data generation.

DIODE (Vasiljevic et al.) ships real dense LiDAR depth (325 indoor + 446 outdoor
validation scans) as ``*.png`` (RGB) + ``*_depth.npy`` (metric float32) +
``*_depth_mask.npy`` (valid pixels).  We turn each view into a self-consistent
fly-through episode:

1.  **Scene** — unproject one view's valid depth into a 3-D coloured point cloud
    (the source camera frame is the *context frame*, origin at the source eye).
2.  **Fly-through** — a randomised smooth Catmull-Rom curve of camera poses that
    keeps the cloud in view (:func:`waveletspace.geometry.flythrough`).
3.  **Noised render** — splat the cloud into a fly-through camera to get a *noised
    grayscale frame* + ground-truth depth/mask (the mesh-plane target).
4.  **Sparse context** — a sparse, noised subsample of the cloud ("the points
    gathered before"), with positional noise *and* a handful of outlier points.

Everything is re-randomised every epoch (:meth:`FlythroughDataset.set_epoch`), and a
synthetic scene generator is used when DIODE is not present so the tests/notebook run
anywhere.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

from . import geometry as G

# Where DIODE may live (the precursor repo ships the val split locally).
_DIODE_CANDIDATES = [
    os.environ.get("DIODE_ROOT", ""),
    "data/diode",
    "../Points_as_supertoroids/data/diode",
    r"C:\work\Points_as_supertoroids\data\diode",
    "/workspace/data/diode",
]


def find_diode_root() -> str | None:
    for c in _DIODE_CANDIDATES:
        if c and os.path.isdir(os.path.join(c, "val")):
            return c
        if c and os.path.isdir(c) and glob.glob(os.path.join(c, "**", "*_depth.npy"), recursive=True):
            return c
    return None


def list_views(root: str) -> list[str]:
    """All DIODE RGB ``.png`` views under ``root`` that have a matching depth file."""
    pngs = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    return [p for p in pngs if os.path.exists(p[:-4] + "_depth.npy")]


def scene_key(path: str) -> str:
    """The physical-scene id for a DIODE view (``scene_NNNNN``), used to split without leakage.

    DIODE packs many near-identical viewpoints of the *same room* into one scene/scan, so a
    flat per-view split would put the same geometry in both train and val.  Grouping by scene
    keeps held-out metrics honest.  Falls back to the file's parent directory name.
    """
    import re
    m = re.search(r"scene_\d+", path.replace("\\", "/"))
    return m.group(0) if m else os.path.basename(os.path.dirname(path))


def grouped_view_split(views, val_frac: float, rng, cap: int = 0):
    """Split ``views`` into ``(train_views, val_views)`` by whole scene (no scene in both).

    ``rng`` shuffles the unique scene keys; ``ceil(n_scenes*val_frac)`` (>=1, and at least one
    scene left for train) become val.  ``cap`` (if >0) limits the number of *train* views after
    the split.  Returns ``(tr_views, vl_views, info)`` where ``info`` reports scene counts.
    """
    import math
    keys = list(dict.fromkeys(scene_key(v) for v in views))     # unique, order-preserving
    rng.shuffle(keys)
    n_val = min(max(1, math.ceil(len(keys) * val_frac)), max(1, len(keys) - 1))
    val_keys = set(keys[:n_val])
    vl = [v for v in views if scene_key(v) in val_keys]
    tr = [v for v in views if scene_key(v) not in val_keys]
    if cap:
        tr = tr[:cap]
    info = {"n_scenes": len(keys), "n_val_scenes": n_val,
            "n_train_scenes": len(keys) - n_val, "n_train_views": len(tr), "n_val_views": len(vl)}
    return tr, vl, info


def load_view(png: str):
    """``png`` -> ``(gray (H,W) in [0,1], depth (H,W) m, mask (H,W) in {0,1})``."""
    from PIL import Image
    rgb = np.asarray(Image.open(png).convert("RGB"), np.float32) / 255.0
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    depth = np.load(png[:-4] + "_depth.npy").astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    mpath = png[:-4] + "_depth_mask.npy"
    mask = np.load(mpath).astype(np.float32) if os.path.exists(mpath) else (depth > 0).astype(np.float32)
    return gray, depth, mask


class Scene:
    """A coloured point cloud in the source-camera (context) frame + bookkeeping."""

    def __init__(self, P, gray, K, name=""):
        self.P = P.astype(np.float32)
        self.gray = gray.astype(np.float32)
        self.K = K
        self.name = name
        self.centroid = self.P.mean(0)
        self.extent = float(np.linalg.norm(self.P - self.centroid, axis=1).std() + 1e-6) * 2.0

    def subsample(self, n, rng):
        if len(self.P) <= n:
            return self.P, self.gray
        idx = rng.choice(len(self.P), n, replace=False)
        return self.P[idx], self.gray[idx]


def scene_from_view(png, vfov=60.0, max_pts=60000, rng=None):
    """Build a :class:`Scene` from one DIODE view (depth clipped to a robust far range)."""
    rng = rng or np.random.default_rng(0)
    gray, depth, mask = load_view(png)
    H, W = depth.shape
    K = G.intrinsics(H, W, vfov)
    valid = (mask > 0.5) & np.isfinite(depth) & (depth > 0)
    if valid.sum() > 0:                                  # clip absurd far returns (sky / windows)
        far = np.percentile(depth[valid], 98)
        valid &= depth <= max(far, 1.0)
    P, idx = G.unproject_depth(depth, K, mask=valid.astype(np.float32))
    g = gray[idx[:, 0], idx[:, 1]]
    if len(P) > max_pts:
        sel = rng.choice(len(P), max_pts, replace=False)
        P, g = P[sel], g[sel]
    return Scene(P, g, K, name=os.path.basename(png))


def synthetic_scene(rng, n=40000, vfov=60.0, H=256, W=256):
    """A DIODE-free fallback scene: a textured wavy surface unprojected to a cloud."""
    K = G.intrinsics(H, W, vfov)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    base = rng.uniform(3.0, 6.0)
    fx, fy = rng.uniform(1.5, 4.0, 2)
    depth = (base + 0.8 * np.sin(uu / W * fx * 2 * np.pi) * np.cos(vv / H * fy * 2 * np.pi)
             + rng.normal(0, 0.05, (H, W)))
    gray = (0.5 + 0.5 * np.sin(uu / W * 8 * np.pi) * np.sin(vv / H * 8 * np.pi)).astype(np.float32)
    P, idx = G.unproject_depth(depth, K)
    g = gray[idx[:, 0], idx[:, 1]]
    if len(P) > n:
        sel = rng.choice(len(P), n, replace=False); P, g = P[sel], g[sel]
    return Scene(P, g, K, name="synthetic")


def noised_render(scene, R, t, img_hw, plane_res, rng, *, vfov=60.0,
                  intensity_noise=0.05, dropout=0.1, radius=1):
    """Render one fly-through frame: a noised grayscale image (``img_hw``) + GT depth/mask
    (``plane_res``).  Returns ``(gray (img_hw,img_hw), depth (plane_res,plane_res),
    mask (plane_res,plane_res), K_plane)``."""
    Ki = G.intrinsics(img_hw, img_hw, vfov)
    Kp = G.intrinsics(plane_res, plane_res, vfov)
    gray, _, gmask = G.splat_render(scene.P, scene.gray, Ki, R, t, img_hw, img_hw, radius=radius)
    # noised grayscale: per-pixel gaussian noise + random dropout of splatted pixels
    gray = gray + rng.normal(0, intensity_noise, gray.shape).astype(np.float32)
    if dropout > 0:
        drop = rng.random(gray.shape) < dropout
        gray[drop] = 0.0
    gray = np.clip(gray, 0.0, 1.0)
    _, depth, mask = G.splat_render(scene.P, scene.gray, Kp, R, t, plane_res, plane_res, radius=radius)
    return gray.astype(np.float32), depth.astype(np.float32), mask.astype(np.float32), Kp


def sample_context(scene, n_pts, rng, *, noise_frac=0.10, n_outliers=10):
    """Sparse, *noised* context points ("the cloud gathered before").

    Draws ``n_pts`` points from the scene, perturbs them by Gaussian noise of std
    ``noise_frac * extent``, then replaces ``n_outliers`` of them with random points in
    the scene's bounding box — covering both readings of "10 noise in the context"
    (a 10% noise level *and* ten spurious points).  Returns ``(n_pts, 3)`` float32.
    """
    P, _ = scene.subsample(n_pts, rng)
    if len(P) < n_pts:                                   # pad by resampling with replacement
        extra = rng.choice(len(P), n_pts - len(P), replace=True)
        P = np.concatenate([P, P[extra]], 0)
    P = P + rng.normal(0, noise_frac * scene.extent, P.shape).astype(np.float32)
    if n_outliers > 0:
        lo, hi = scene.P.min(0), scene.P.max(0)
        k = min(n_outliers, n_pts)
        oi = rng.choice(n_pts, k, replace=False)
        P[oi] = rng.uniform(lo, hi, (k, 3)).astype(np.float32)
    return P.astype(np.float32)


def make_episode(scene, rng, *, img_hw=256, plane_res=64, n_ctx_points=512,
                 vfov=60.0, ctx_noise=0.10, n_outliers=10, n_ctrl=4, move_frac=0.25,
                 chamfer_pts=4096, radius=1):
    """One training/eval episode dict from a scene: render a random fly-through frame,
    sample noised context, and package GT pose + depth + a chamfer reference cloud."""
    targets, _ = scene.subsample(96, rng)                # look-at candidates spanning the scene
    Rs, ts = G.flythrough(scene.centroid, scene.extent, rng, n_frames=8,
                          n_ctrl=n_ctrl, targets=targets, move_frac=move_frac)
    # the explore path dollies in/out and pans across the scene, so a few viewpoints frame
    # little surface; probe candidates at low res and keep the best-filled one (usable target).
    best_f, best_fill = int(rng.integers(0, len(Rs))), -1.0
    Kprobe = G.intrinsics(48, 48, vfov)
    for _ in range(5):
        f = int(rng.integers(0, len(Rs)))
        _, _, pm = G.splat_render(scene.P, scene.gray, Kprobe, Rs[f], ts[f], 48, 48, radius=1)
        fill = float(pm.mean())
        if fill > best_fill:
            best_f, best_fill = f, fill
        if fill > 0.12:
            break
    R, t = Rs[best_f], ts[best_f]
    gray, depth, mask, Kp = noised_render(scene, R, t, img_hw, plane_res, rng, vfov=vfov, radius=radius)
    ctx = sample_context(scene, n_ctx_points, rng, noise_frac=ctx_noise, n_outliers=n_outliers)
    sp, _ = scene.subsample(chamfer_pts, rng)
    return {
        "img": torch.from_numpy(gray)[None],            # (1,H,W)
        "ctx": torch.from_numpy(ctx),                   # (Nc,3)
        "depth": torch.from_numpy(depth)[None],         # (1,plane_res,plane_res)
        "mask": torch.from_numpy(mask)[None],           # (1,plane_res,plane_res)
        "R": torch.from_numpy(R.astype(np.float32)),    # (3,3) cam->scene
        "t": torch.from_numpy(t.astype(np.float32)),    # (3,)
        "K_plane": torch.from_numpy(Kp.astype(np.float32)),
        "scene_pts": torch.from_numpy(sp.astype(np.float32)),
        "name": scene.name,
    }


class FlythroughDataset(torch.utils.data.Dataset):
    """Per-epoch-randomised DIODE fly-throughs (falls back to synthetic scenes).

    Each ``__getitem__`` builds a *fresh* episode seeded by ``(seed, epoch, index)`` so
    every epoch sees new curves, renders, noise and context — call
    :meth:`set_epoch` once per epoch.  When ``root`` is ``None`` and DIODE is not found,
    synthetic scenes are generated so the pipeline runs with no data.
    """

    def __init__(self, root: str | None = "auto", *, img_hw=256, plane_res=64,
                 n_ctx_points=512, vfov=60.0, ctx_noise=0.10, n_outliers=10,
                 max_scene_pts=60000, chamfer_pts=4096, length=None, seed=0,
                 synthetic=False, move_frac=0.25, views=None, radius=1):
        self.cfg = dict(img_hw=img_hw, plane_res=plane_res, n_ctx_points=n_ctx_points,
                        vfov=vfov, ctx_noise=ctx_noise, n_outliers=n_outliers,
                        chamfer_pts=chamfer_pts, move_frac=move_frac, radius=radius)
        self.max_scene_pts = max_scene_pts
        self.seed = seed
        self.epoch = 0
        self.synthetic = synthetic
        self.root = None if synthetic else (find_diode_root() if root == "auto" else root)
        if views is not None:                            # explicit view list (e.g. a train/val split)
            self.views = list(views)
        else:
            self.views = list_views(self.root) if self.root else []
        if not self.views:
            self.synthetic = True
        self._len = length or (len(self.views) if self.views else 256)

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def __len__(self):
        return self._len

    def _scene(self, i, rng):
        if self.synthetic or not self.views:
            return synthetic_scene(rng)
        png = self.views[i % len(self.views)]
        return scene_from_view(png, vfov=self.cfg["vfov"], max_pts=self.max_scene_pts, rng=rng)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 9973 + i) & 0x7FFFFFFF)
        scene = self._scene(i, rng)
        return make_episode(scene, rng, **self.cfg)


def collate(batch):
    """Stack a list of episode dicts; pad ``scene_pts`` to the batch-max for chamfer."""
    out = {}
    for k in ("img", "ctx", "depth", "mask", "R", "t", "K_plane"):
        out[k] = torch.stack([b[k] for b in batch], 0)
    smax = max(b["scene_pts"].shape[0] for b in batch)
    sp = torch.zeros(len(batch), smax, 3)
    spn = torch.zeros(len(batch), dtype=torch.long)
    for j, b in enumerate(batch):
        n = b["scene_pts"].shape[0]; sp[j, :n] = b["scene_pts"]; spn[j] = n
    out["scene_pts"] = sp; out["scene_n"] = spn
    out["name"] = [b["name"] for b in batch]
    return out
