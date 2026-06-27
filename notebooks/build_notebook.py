"""Generate notebooks/waveletspace_colab.ipynb (run:  python notebooks/build_notebook.py).

Keeping the notebook in a small builder (instead of hand-editing JSON) is the same
pattern the precursor repo uses — the .ipynb is a build artifact of this file.

This notebook is tuned for a **Colab A100 40 GB** runtime: it downloads the *complete*
DIODE **train** set (~87 GB, streamed) and trains the *complete* model (full
1024→32 wavelet pyramid).  Weights + logs are persisted to Google Drive.
"""
import json
import os

REPO = "OlegJakushkin/WaveletSpaceNet"
REPO_URL = f"https://github.com/{REPO}.git"
DIODE_TRAIN = "https://diode-dataset.s3.amazonaws.com/train.tar.gz"   # ~87 GB
DIODE_VAL = "https://diode-dataset.s3.amazonaws.com/val.tar.gz"       # ~2.8 GB
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/waveletspace_colab.ipynb"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
DRIVE_DIR = "/content/drive/MyDrive/WaveletSpaceNet"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(lines)}


def _lines(lines):
    return ("\n".join(lines)).splitlines(keepends=True) or [""]


CELLS = [
    md(f"# WaveletSpaceNet — full training on the complete DIODE train set (A100 40 GB)",
       "",
       f"[![Open In Colab]({BADGE})]({COLAB})",
       "",
       "Trains the **complete** model — the full `1024,512,256,128,64,32` wavelet image pyramid — on the",
       "**complete DIODE train set** (~87 GB, ~25 k LiDAR views), then runs inference (a grayscale frame +",
       "optional sparse 3-D context → a **mesh-plane** + **camera pose**).  Weights + logs are saved to",
       f"Google Drive (`{DRIVE_DIR}`).",
       "",
       "**Requirements:** *Runtime → Change runtime type → **A100 GPU** + High-RAM* (Colab Pro+).  You need",
       "~95 GB of free disk for the train set and several hours for a full run.  Short on resources?  Set",
       "`FULL_TRAIN = False` in the data cell to use the 2.8 GB validation split instead."),

    md("## 0 · Check the GPU (expecting A100 40 GB)"),
    code("import torch",
         "!nvidia-smi -L || echo 'no GPU detected'",
         "if torch.cuda.is_available():",
         "    name = torch.cuda.get_device_name(0)",
         "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9",
         "    print(f'GPU: {name} | {vram:.0f} GB VRAM')",
         "    if 'A100' not in name:",
         "        print('!! Tuned for an A100 40 GB. On a smaller GPU, lower --batch / --img-hw / --levels in the train cell.')",
         "else:",
         "    print('!! No GPU — set Runtime → Change runtime type → A100 GPU (High-RAM).')"),

    md("## 1 · Clone the repository (idempotent — force-syncs to the latest `main`)",
       "",
       "Re-running this always brings the working tree to the newest commit (even if a stale clone was",
       "left from an earlier session) and drops any already-imported `waveletspace` modules so the rest of",
       "the notebook picks up the update without a kernel restart."),
    code("import os, sys, subprocess",
         "if os.path.basename(os.getcwd()) != 'WaveletSpaceNet':",
         "    if not os.path.isdir('WaveletSpaceNet'):",
         f"        !git clone {REPO_URL}",
         "    %cd WaveletSpaceNet",
         "# force the working tree to exactly match the latest origin/main (robust to a stale clone)",
         "!git fetch -q origin && git reset --hard origin/main",
         "# drop any already-imported waveletspace modules so later cells re-read the updated code",
         "for _m in [x for x in list(sys.modules) if x == 'waveletspace' or x.startswith('waveletspace.')]:",
         "    del sys.modules[_m]",
         "print('cwd:', os.getcwd(), '| HEAD', subprocess.getoutput('git rev-parse --short HEAD'))"),

    md("## 2 · Install dependencies",
       "",
       "Colab already ships a compatible `torch`/`numpy`/`matplotlib`/`pillow`, so this only fills any",
       "gaps (it will not reinstall or downgrade torch)."),
    code("!pip -q install -r requirements.txt",
         "print('deps ready')"),

    md("## 3 · Mount Google Drive (persist weights + logs)",
       "",
       f"Links the local `assets/` (weights) and `renders/` (logs + panels) to `{DRIVE_DIR}`, so **every**",
       "checkpoint is written straight to Drive during training and survives the runtime being recycled.",
       "The 87 GB DIODE train set is **not** cached to Drive (too large) — it lives on the Colab disk."),
    code("import os, shutil",
         "WS_DRIVE = None",
         "try:",
         "    from google.colab import drive",
         "    drive.mount('/content/drive')",
         f"    WS_DRIVE = '{DRIVE_DIR}'",
         "    for sub in ['data', 'assets', 'renders', 'runs']:",
         "        os.makedirs(f'{WS_DRIVE}/{sub}', exist_ok=True)",
         "    for d in ['assets', 'renders']:",
         "        tgt = f'{WS_DRIVE}/{d}'",
         "        if os.path.islink(d):",
         "            continue",
         "        if os.path.isdir(d):",
         "            for f in os.listdir(d):",
         "                shutil.move(os.path.join(d, f), os.path.join(tgt, f))",
         "            os.rmdir(d)",
         "        os.symlink(tgt, d)",
         "    print('Drive mounted -> weights:', f'{WS_DRIVE}/assets', '| logs:', f'{WS_DRIVE}/renders')",
         "except Exception as e:",
         "    print('Drive not mounted (not on Colab?):', e, '\\n-> everything stays local')"),

    md("## 4 · Download the complete DIODE train set (~87 GB)",
       "",
       "The train set is **streamed and extracted on the fly** (no 87 GB tarball is kept on disk, so peak",
       "usage ≈ the extracted size).  Expect ~20–60 min depending on bandwidth; there is no progress bar.",
       "If the stream is interrupted, run `!rm -rf data/diode/train` and re-run this cell.",
       "",
       "Set `FULL_TRAIN = False` to instead grab the small (2.8 GB) **val** split for a quick end-to-end test."),
    code("FULL_TRAIN = True   # True = complete DIODE train set (~87 GB). False = val split (~2.8 GB).",
         "import os, glob, shutil, tarfile",
         "os.makedirs('data/diode', exist_ok=True)",
         "have_train = bool(glob.glob('data/diode/train/**/*_depth.npy', recursive=True))",
         "have_val = bool(glob.glob('data/diode/val/**/*_depth.npy', recursive=True))",
         "free_gb = shutil.disk_usage('/content').free / 1e9",
         "print(f'free disk: {free_gb:.0f} GB')",
         "if FULL_TRAIN and not have_train:",
         "    if free_gb < 95:",
         "        print('!! < 95 GB free — the train set may not fit; consider FULL_TRAIN = False.')",
         "    print('streaming + extracting DIODE train (~87 GB, ~20-60 min, no progress bar)...')",
         f"    !wget -qO- {DIODE_TRAIN} | tar xz -C data/diode",
         "elif (not FULL_TRAIN) and not have_val:",
         "    drive_tar = f'{WS_DRIVE}/data/val.tar.gz' if WS_DRIVE else None",
         "    if drive_tar and os.path.exists(drive_tar):",
         "        print('using the DIODE val tarball cached on Drive'); tarball = drive_tar",
         "    else:",
         f"        !wget -c -q --show-progress {DIODE_VAL} -O val.tar.gz",
         "        tarball = 'val.tar.gz'",
         "        if drive_tar:",
         "            print('caching val tarball to Drive...'); shutil.copy('val.tar.gz', drive_tar)",
         "    try:",
         "        with tarfile.open(tarball) as tf: tf.extractall('data/diode', filter='data')",
         "    except TypeError:",
         "        with tarfile.open(tarball) as tf: tf.extractall('data/diode')",
         "    if tarball == 'val.tar.gz' and os.path.exists('val.tar.gz'): os.remove('val.tar.gz')",
         "n = len(glob.glob('data/diode/**/*_depth.npy', recursive=True))",
         "print('DIODE depth views found:', n, '(0 → training will use synthetic scenes)')"),

    md("## 5 · Tests — a single fly-through generates + trains locally"),
    code("!python -m pytest tests/ -q"),

    md("## 6 · Visualise a generated fly-through",
       "",
       "A random smooth Catmull-Rom curve that **explores** the scene — dollying closer / pulling back and",
       "panning to look at different parts (the look-at sweeps actual scene points).  Top row = noised",
       "grayscale renders, bottom row = ground-truth depth; titles show the camera distance."),
    code("import numpy as np, matplotlib.pyplot as plt",
         "from waveletspace import diode as D, geometry as G",
         "root = D.find_diode_root()",
         "rng = np.random.default_rng(0)",
         "scene = D.scene_from_view(D.list_views(root)[0], max_pts=150000) if root else D.synthetic_scene(rng)",
         "targets, _ = scene.subsample(96, rng)",
         "Rs, ts = G.flythrough(scene.centroid, scene.extent, rng, n_frames=8, targets=targets)",
         "fig, ax = plt.subplots(2, 8, figsize=(20, 5.5))",
         "for i in range(8):",
         "    g, d, m, _ = D.noised_render(scene, Rs[i], ts[i], 256, 64, rng, radius=2)",
         "    dist = np.linalg.norm(ts[i] - scene.centroid)",
         "    ax[0, i].imshow(g, cmap='gray'); ax[0, i].set_title(f'{dist:.1f} m'); ax[0, i].axis('off')",
         "    ax[1, i].imshow(np.where(d > 0, d, np.nan), cmap='turbo'); ax[1, i].axis('off')",
         "plt.tight_layout(); plt.show()",
         "dd = np.linalg.norm(ts - scene.centroid, axis=1)",
         "print('camera distance range: %.1f-%.1f m (dolly) | scene: %s' % (dd.min(), dd.max(), scene.name))",
         "print('sparse noised context:', D.sample_context(scene, 512, rng).shape)"),

    md("## 7 · Save a snapshot of the generated training data to Drive"),
    code("import numpy as np",
         "from waveletspace import diode as D",
         "ds = D.FlythroughDataset('auto', img_hw=256, plane_res=64, n_ctx_points=512, max_scene_pts=150000, radius=2)",
         "ds.set_epoch(0)",
         "N = min(16, len(ds)); samples = [ds[i] for i in range(N)]",
         "dest = (f'{WS_DRIVE}/data/flythrough_samples.npz' if WS_DRIVE else 'data/flythrough_samples.npz')",
         "np.savez_compressed(dest,",
         "    img=np.stack([s['img'].numpy() for s in samples]),",
         "    depth=np.stack([s['depth'].numpy() for s in samples]),",
         "    mask=np.stack([s['mask'].numpy() for s in samples]),",
         "    ctx=np.stack([s['ctx'].numpy() for s in samples]),",
         "    R=np.stack([s['R'].numpy() for s in samples]),",
         "    t=np.stack([s['t'].numpy() for s in samples]),",
         "    names=np.array([s['name'] for s in samples]))",
         "print(f'saved {N} training episodes ->', dest)"),

    md("## 8 · Train the complete model on the complete train set (A100)",
       "",
       "Full `1024→32` wavelet pyramid, `plane_res=128` mesh-plane, on the held-out-by-scene split of the",
       "DIODE train set.  Checkpoints (`waveletspace_full.pt` / `_latest.pt`) and render panels are written",
       "to `assets/` / `renders/`, which are **linked to Drive**, so they persist + you can resume.",
       "",
       "This is a multi-hour run.  Reduce `--batch` if you hit OOM; resume an interrupted run by adding",
       "`--resume assets/waveletspace_full_latest.pt`."),
    code("!python train.py --epochs 30 --batch 16 --img-hw 1024 --plane-res 128 --levels 1024,512,256,128,64,32 --d 256 --M 256 --L 6 --workers 8 --max-scene-pts 150000 --splat-radius 2 --val-cap 256 --out waveletspace_full"),

    md("## 9 · Training curve + render panel"),
    code("import json, glob",
         "from IPython.display import Image as IPImage, display",
         "hist = json.load(open('renders/ws_train_hist.json'))",
         "print('val chamfer(m) per epoch:', [round(h['val']['chamfer'], 3) for h in hist])",
         "print('val logdepthL1 per epoch:', [round(h['val']['logdepthL1'], 3) for h in hist])",
         "panels = sorted(glob.glob('renders/ws_val_ep*.png'))",
         "if panels: display(IPImage(panels[-1]))"),

    md("## 10 · Inference → mesh-plane + camera pose"),
    code("import os, glob, numpy as np, torch, matplotlib.pyplot as plt",
         "from waveletspace.model import load_checkpoint, WaveletSpaceNet",
         "from waveletspace import diode as D",
         "from waveletspace.infer_helpers import mesh_plane_verts",
         "cands = ['assets/waveletspace_full.pt', 'assets/waveletspace.pt'] + sorted(glob.glob('assets/*.pt'))",
         "ckpt = next((c for c in cands if os.path.exists(c)), None)",
         "dev = 'cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'",
         "if ckpt:",
         "    net, _ = load_checkpoint(ckpt, map_location='cpu'); print('loaded', ckpt)",
         "else:",
         "    net = WaveletSpaceNet(levels=(256,128,64,32), plane_res=64, d=96, M=64, L=3, heads=4); print('untrained model')",
         "net = net.to(dev).eval()",
         "ds = D.FlythroughDataset('auto', img_hw=net.img_tok.top, plane_res=net.plane_res, max_scene_pts=150000, radius=2)",
         "ds.set_epoch(3); ep = ds[0]",
         "with torch.no_grad():",
         "    out = net(ep['img'][None].to(dev), ep['ctx'][None].to(dev))",
         "verts, _ = mesh_plane_verts(out)",
         "print('pred camera t', np.round(out['t'][0].cpu().numpy(), 3), '| GT t', np.round(ep['t'].numpy(), 3))",
         "fig = plt.figure(figsize=(16, 4))",
         "a = fig.add_subplot(1, 4, 1); a.imshow(ep['img'][0], cmap='gray'); a.set_title('input frame'); a.axis('off')",
         "gt = ep['depth'][0].numpy(); a = fig.add_subplot(1, 4, 2); a.imshow(np.where(gt > 0, gt, np.nan), cmap='turbo'); a.set_title('GT depth'); a.axis('off')",
         "a = fig.add_subplot(1, 4, 3); a.imshow(out['depth'][0, 0].cpu().numpy(), cmap='turbo'); a.set_title('pred depth'); a.axis('off')",
         "a = fig.add_subplot(1, 4, 4, projection='3d'); s = verts[::8]",
         "a.scatter(s[:, 0], s[:, 2], -s[:, 1], c=s[:, 2], s=2, cmap='turbo'); a.set_title('mesh-plane (3D)')",
         "plt.tight_layout(); plt.show()"),

    md("## 11 · Export the mesh-plane (.obj) → Drive + download"),
    code("import os, glob, shutil",
         "ck = next((c for c in ['assets/waveletspace_full.pt', 'assets/waveletspace.pt'] if os.path.exists(c)), '')",
         "!python infer.py --demo --ckpt {ck} --out out/plane.obj",
         "if WS_DRIVE and os.path.exists('out/plane.obj'):",
         "    shutil.copy('out/plane.obj', f'{WS_DRIVE}/runs/plane.obj'); print('copied to', f'{WS_DRIVE}/runs/plane.obj')",
         "try:",
         "    from google.colab import files; files.download('out/plane.obj')",
         "except Exception as e:",
         "    print('download skipped (', e, ') — find it at out/plane.obj')"),

    md("## 12 · What was saved to Drive"),
    code("import glob",
         "if WS_DRIVE:",
         "    print('weights:'); _ = [print('  ', p) for p in glob.glob(f'{WS_DRIVE}/assets/*.pt')]",
         "    print('logs/panels:', len(glob.glob(f'{WS_DRIVE}/renders/*')), 'files')",
         "    print('training data:'); _ = [print('  ', p) for p in glob.glob(f'{WS_DRIVE}/data/*')]",
         "    print('runs:'); _ = [print('  ', p) for p in glob.glob(f'{WS_DRIVE}/runs/*')]",
         "else:",
         "    print('Drive was not mounted — artifacts are local: assets/, renders/, out/, data/')"),
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
