"""Generate notebooks/waveletspace_colab.ipynb (run:  python notebooks/build_notebook.py).

Keeping the notebook in a small builder (instead of hand-editing JSON) is the same
pattern the precursor repo uses — the .ipynb is a build artifact of this file.
The notebook is written to run top-to-bottom in Google Colab on a GPU runtime.
"""
import json
import os

REPO = "OlegJakushkin/WaveletSpaceNet"
REPO_URL = f"https://github.com/{REPO}.git"
DIODE_URL = "https://diode-dataset.s3.amazonaws.com/val.tar.gz"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/waveletspace_colab.ipynb"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(lines)}


def _lines(lines):
    return ("\n".join(lines)).splitlines(keepends=True) or [""]


CELLS = [
    md(f"# WaveletSpaceNet — sparse context + a wavelet image pyramid → mesh-plane + camera pose",
       "",
       f"[![Open In Colab]({BADGE})]({COLAB})",
       "",
       "Clones the repo, installs it, downloads **DIODE**, runs the tests, visualises a generated",
       "**fly-through**, trains on the Colab GPU, and runs inference (a grayscale frame + optional sparse",
       "3-D context → a **mesh-plane** + **camera pose** relative to the context).",
       "",
       "**Before you start:** *Runtime → Change runtime type → Hardware accelerator = **GPU***, then",
       "*Runtime → Run all*.  Every cell is idempotent (safe to re-run)."),

    md("## 0 · Check the GPU"),
    code("import torch",
         "!nvidia-smi -L || echo 'no GPU — switch the runtime to GPU for training'",
         "print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"),

    md("## 1 · Clone the repository (idempotent)"),
    code("import os",
         "if os.path.basename(os.getcwd()) != 'WaveletSpaceNet':",
         "    if not os.path.isdir('WaveletSpaceNet'):",
         f"        !git clone {REPO_URL}",
         "    %cd WaveletSpaceNet",
         "!git pull --ff-only || true",
         "print('cwd:', os.getcwd())"),

    md("## 2 · Install dependencies",
       "",
       "Colab already ships a compatible `torch`/`numpy`/`matplotlib`/`pillow`, so this only fills any",
       "gaps (it will not reinstall or downgrade torch)."),
    code("!pip -q install -r requirements.txt",
         "print('deps ready')"),

    md("## 3 · Download DIODE (validation split: 325 indoor + 446 outdoor)",
       "",
       "≈ 2.8 GB; skipped if already present.  If you skip this, the tests and training fall back to a",
       "synthetic scene generator so everything still runs."),
    code("import os, glob",
         "os.makedirs('data/diode', exist_ok=True)",
         "if not glob.glob('data/diode/**/*_depth.npy', recursive=True):",
         f"    !wget -c -q --show-progress {DIODE_URL} -O val.tar.gz",
         "    !tar xf val.tar.gz -C data/diode && rm -f val.tar.gz",
         "n = len(glob.glob('data/diode/**/*_depth.npy', recursive=True))",
         "print('DIODE depth views found:', n, '(0 → training will use synthetic scenes)')"),

    md("## 4 · Tests — a single fly-through generates + trains locally"),
    code("!python -m pytest tests/ -q"),

    md("## 5 · Visualise a generated fly-through",
       "",
       "A random smooth Catmull-Rom curve → noised grayscale renders + ground-truth depth, plus the",
       "sparse noised context cloud ('the points gathered before')."),
    code("import numpy as np, matplotlib.pyplot as plt",
         "from waveletspace import diode as D, geometry as G",
         "root = D.find_diode_root()",
         "rng = np.random.default_rng(0)",
         "scene = D.scene_from_view(D.list_views(root)[0]) if root else D.synthetic_scene(rng)",
         "Rs, ts = G.flythrough(scene.centroid, scene.extent, rng, n_frames=6)",
         "fig, ax = plt.subplots(2, 6, figsize=(18, 6))",
         "for i in range(6):",
         "    g, d, m, _ = D.noised_render(scene, Rs[i], ts[i], 256, 64, rng)",
         "    ax[0, i].imshow(g, cmap='gray'); ax[0, i].set_title(f'frame {i}'); ax[0, i].axis('off')",
         "    ax[1, i].imshow(np.where(d > 0, d, np.nan), cmap='turbo'); ax[1, i].axis('off')",
         "ax[0, 0].set_ylabel('noised gray'); ax[1, 0].set_ylabel('GT depth')",
         "plt.tight_layout(); plt.show()",
         "ctx = D.sample_context(scene, 512, rng)",
         "print('sparse noised context:', ctx.shape, '| scene:', scene.name)"),

    md("## 6 · Train on the Colab GPU",
       "",
       "A short demo run (top pyramid level = 256 to fit a Colab T4 and finish quickly).  The model's",
       "**default** is the full `1024,512,256,128,64,32` pyramid — on a large-memory GPU run the",
       "commented line instead.  `--workers 2` parallelises the on-the-fly fly-through rendering."),
    code("!python train.py --epochs 8 --batch 8 --img-hw 256 --plane-res 64 \\",
         "    --levels 256,128,64,32 --d 256 --M 256 --L 6 --workers 2 --out waveletspace",
         "",
         "# full pyramid (needs a big GPU, much slower):",
         "# !python train.py --epochs 40 --batch 4 --img-hw 1024 --plane-res 64 \\",
         "#     --levels 1024,512,256,128,64,32 --workers 2 --out waveletspace"),

    md("## 7 · Training curve + render panel"),
    code("import json, glob",
         "from IPython.display import Image as IPImage, display",
         "hist = json.load(open('renders/ws_train_hist.json'))",
         "print('val chamfer(m) per epoch:', [round(h['val']['chamfer'], 3) for h in hist])",
         "print('val logdepthL1 per epoch:', [round(h['val']['logdepthL1'], 3) for h in hist])",
         "panels = sorted(glob.glob('renders/ws_val_ep*.png'))",
         "if panels: display(IPImage(panels[-1]))"),

    md("## 8 · Inference → mesh-plane + camera pose",
       "",
       "Input grayscale frame (+ optional sparse context) → predicted depth and the mesh-plane placed",
       "in the context frame by the predicted pose."),
    code("import os, glob, numpy as np, torch, matplotlib.pyplot as plt",
         "from waveletspace.model import load_checkpoint, WaveletSpaceNet",
         "from waveletspace import diode as D",
         "from waveletspace.infer_helpers import mesh_plane_verts",
         "ckpt = 'assets/waveletspace.pt' if os.path.exists('assets/waveletspace.pt') else (glob.glob('assets/*.pt') + [None])[0]",
         "dev = 'cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'",
         "if ckpt:",
         "    net, _ = load_checkpoint(ckpt, map_location='cpu'); print('loaded', ckpt)",
         "else:",
         "    net = WaveletSpaceNet(levels=(256,128,64,32), plane_res=64, d=96, M=64, L=3, heads=4); print('untrained model')",
         "net = net.to(dev).eval()",
         "ds = D.FlythroughDataset('auto', img_hw=net.img_tok.top, plane_res=net.plane_res); ds.set_epoch(3)",
         "ep = ds[0]",
         "with torch.no_grad():",
         "    out = net(ep['img'][None].to(dev), ep['ctx'][None].to(dev))",
         "verts, _ = mesh_plane_verts(out)",
         "print('pred camera t', np.round(out['t'][0].cpu().numpy(), 3), '| GT t', np.round(ep['t'].numpy(), 3))",
         "fig = plt.figure(figsize=(16, 4))",
         "a = fig.add_subplot(1, 4, 1); a.imshow(ep['img'][0], cmap='gray'); a.set_title('input frame'); a.axis('off')",
         "gt = ep['depth'][0].numpy(); a = fig.add_subplot(1, 4, 2); a.imshow(np.where(gt > 0, gt, np.nan), cmap='turbo'); a.set_title('GT depth'); a.axis('off')",
         "a = fig.add_subplot(1, 4, 3); a.imshow(out['depth'][0, 0].cpu().numpy(), cmap='turbo'); a.set_title('pred depth'); a.axis('off')",
         "a = fig.add_subplot(1, 4, 4, projection='3d'); s = verts[::4]",
         "a.scatter(s[:, 0], s[:, 2], -s[:, 1], c=s[:, 2], s=2, cmap='turbo'); a.set_title('mesh-plane (3D)')",
         "plt.tight_layout(); plt.show()"),

    md("## 9 · Export the mesh-plane (.obj)"),
    code("!python infer.py --demo --ckpt assets/waveletspace.pt --out out/plane.obj",
         "try:",
         "    from google.colab import files; files.download('out/plane.obj')",
         "except Exception as e:",
         "    print('download skipped (', e, ') — find it at out/plane.obj')"),
]


def main():
    for i, c in enumerate(CELLS):                     # stable cell ids (nbformat 4.5+)
        c["id"] = f"cell{i:02d}"
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "waveletspace_colab.ipynb", "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(__file__), "waveletspace_colab.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, "(%d cells)" % len(CELLS))
    print("Colab URL:", COLAB)


if __name__ == "__main__":
    main()
