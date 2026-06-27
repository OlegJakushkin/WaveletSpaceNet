"""Turn a model output into a mesh-plane (grid mesh) + .obj export."""
from __future__ import annotations

import os
import numpy as np
import torch

from . import geometry as G


def mesh_plane_verts(out: dict, vfov: float = 60.0):
    """Unproject predicted depth into the context frame using the predicted pose.

    Returns ``(verts (Q,3) np.float32, faces (F,3) np.int64)`` for an ``r×r`` grid mesh,
    where ``r`` is the predicted depth resolution.
    """
    r = out["depth"].shape[-1]
    K = torch.from_numpy(G.intrinsics(r, r, vfov).astype(np.float32))[None].to(out["depth"].device)
    verts = G.depth_to_points(out["depth"], K, out["R"], out["t"])[0].detach().cpu().numpy()
    faces = G.grid_faces(r, r)
    return verts.astype(np.float32), faces


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for tri in faces + 1:                       # OBJ is 1-indexed
            f.write(f"f {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")
