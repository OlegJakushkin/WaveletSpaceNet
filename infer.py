"""Inference: a grayscale frame (+ optional sparse context) -> mesh-plane .obj + camera pose.

  python infer.py --image frame.png --ckpt assets/waveletspace.pt --out out/plane.obj
  python infer.py --image frame.png --context ctx.npy --out out/plane.obj
  python infer.py --demo                                  # run one DIODE/synthetic fly-through frame

``--context`` is a ``.npy`` of shape ``(N, 3)`` sparse 3-D points in the context frame
(omit for the empty-context case).  Writes the unprojected mesh-plane (placed in the
context frame by the predicted pose) and prints the predicted camera pose (R, t).
"""
import argparse, os
import numpy as np, torch

from waveletspace.model import load_checkpoint, WaveletSpaceNet
from waveletspace import diode as D
from waveletspace.infer_helpers import mesh_plane_verts, write_obj


def load_gray(path, hw):
    from PIL import Image
    im = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    t = torch.from_numpy(im)[None, None]
    return torch.nn.functional.interpolate(t, size=(hw, hw), mode="area")[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--context", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="assets/waveletspace.pt")
    ap.add_argument("--out", type=str, default="out/plane.obj")
    ap.add_argument("--vfov", type=float, default=60.0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--demo", action="store_true", help="use one generated fly-through frame")
    a = ap.parse_args()
    dev = a.device or ("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

    if os.path.exists(a.ckpt):
        net, _ = load_checkpoint(a.ckpt, map_location="cpu")     # load on CPU, then move (robust)
        print(f"loaded {a.ckpt}")
    else:
        print(f"!! {a.ckpt} not found -> using an UNTRAINED model (geometry demo only)")
        net = WaveletSpaceNet(d=96, M=64, L=3, heads=4, plane_res=64, wave_levels=2, img_size=128)
    net = net.to(dev).eval()
    hw = net.img_tok.top

    if a.demo or not a.image:
        ds = D.FlythroughDataset("auto", img_hw=hw, plane_res=net.plane_res)
        ds.set_epoch(0); ep = ds[0]
        img = ep["img"][None].to(dev); ctx = ep["ctx"][None].to(dev)
        print(f"demo scene: {ep['name']} | GT t {ep['t'].tolist()}")
    else:
        img = load_gray(a.image, hw)[None, None].to(dev)
        ctx = None
        if a.context and os.path.exists(a.context):
            ctx = torch.from_numpy(np.load(a.context).astype(np.float32))[None].to(dev)
            print(f"context: {tuple(ctx.shape)}")

    with torch.no_grad():
        out = net(img, ctx)
    print("predicted camera-to-context pose:")
    print("  R =", np.round(out["R"][0].cpu().numpy(), 4).tolist())
    print("  t =", np.round(out["t"][0].cpu().numpy(), 4).tolist())
    verts, faces = mesh_plane_verts(out, a.vfov)
    write_obj(a.out, verts, faces)
    print(f"mesh-plane -> {a.out}  ({len(verts)} verts, {len(faces)} faces)")


if __name__ == "__main__":
    main()
