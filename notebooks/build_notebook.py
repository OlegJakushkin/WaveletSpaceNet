"""Generate notebooks/waveletspace_colab.ipynb (run:  python notebooks/build_notebook.py).

Keeping the notebook in a small builder (instead of hand-editing JSON) is the same
pattern the precursor repo uses — the .ipynb is a build artifact of this file.
"""
import json
import os

REPO_URL = "https://github.com/OlegJakushkin/WaveletSpaceNet.git"
DIODE_URL = "http://diode-dataset.s3.amazonaws.com/val.tar.gz"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(_lines(lines))}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(_lines(lines))}


def _lines(lines):
    text = "\n".join(lines)
    out = text.splitlines(keepends=True)
    return out or [""]


CELLS = [
    md("# WaveletSpaceNet — sparse context + a wavelet image pyramid → mesh-plane + camera pose",
       "",
       "This notebook clones the repo, installs it, downloads the **DIODE** validation split,",
       "runs the tests, visualises a generated **fly-through**, trains on the Colab GPU and runs",
       "inference (a grayscale frame + optional sparse context → a **mesh-plane** + **camera pose**).",
       "",
       "> Runtime → *Change runtime type* → **GPU** before running the training cell."),

    md("## 1 · Clone the repository"),
    code("import os",
         f"if not os.path.exists('WaveletSpaceNet'):",
         f"    !git clone {REPO_URL}",
         "%cd WaveletSpaceNet",
         "!git pull --ff-only || true"),

    md("## 2 · Install dependencies"),
    code("!pip -q install -r requirements.txt"),

    md("## 3 · Download DIODE (validation: 325 indoor + 446 outdoor)",
       "",
       "~ a few GB; skipped if already present.  The tests and training fall back to a synthetic",
       "scene generator if you skip this cell."),
    code("import os, glob",
         "os.makedirs('data/diode', exist_ok=True)",
         "if not glob.glob('data/diode/**/*_depth.npy', recursive=True):",
         f"    !wget -q {DIODE_URL} -O val.tar.gz",
         "    !tar xf val.tar.gz -C data/diode && rm -f val.tar.gz",
         "print('DIODE depth views:', len(glob.glob('data/diode/**/*_depth.npy', recursive=True)))"),

    md("## 4 · Tests — a single fly-through generates + trains locally"),
    code("!python -m pytest tests/ -q"),

    md("## 5 · Visualise a generated fly-through",
       "",
       "A random smooth curve → noised grayscale renders + ground-truth depth, plus the sparse",
       "noised context cloud ('the points gathered before')."),
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
         "plt.tight_layout(); plt.show()",
         "ctx = D.sample_context(scene, 512, rng)",
         "print('sparse noised context:', ctx.shape, '| scene name:', scene.name)"),

    md("## 6 · Train on the Colab GPU",
       "",
       "The default model uses the full `1024,512,256,128,64,32` wavelet pyramid; that is heavy,",
       "so for a Colab session we cap the top level at 512.  Increase `--levels` / `--epochs` for a",
       "real run."),
    code("!python train.py --epochs 6 --batch 8 --img-hw 256 --plane-res 64 \\",
         "    --levels 512,256,128,64,32 --d 256 --M 256 --L 6 --out waveletspace"),

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
       "`--demo` runs one generated fly-through frame; or pass `--image frame.png "
       "[--context ctx.npy]`."),
    code("!python infer.py --demo --ckpt assets/waveletspace.pt --out out/plane.obj",
         "print('\\nwrote out/plane.obj (download it / open in a mesh viewer)')"),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "waveletspace_colab.ipynb", "provenance": []},
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


if __name__ == "__main__":
    main()
