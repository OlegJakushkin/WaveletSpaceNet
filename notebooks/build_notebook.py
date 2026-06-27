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
    md(f"# WaveletSpaceNet v2 — full training on the complete DIODE train set (A100 40 GB)",
       "",
       f"[![Open In Colab]({BADGE})]({COLAB})",
       "",
       "Trains the **v2 \"Sharp-FPN\"** model — a learned ConvNeXt-lite + FPN image embedding (grayscale +",
       "computed edge channels, sensor-agnostic) feeding a Perceiver, with a **multi-level Haar** mesh-plane",
       "decoder at `plane_res=256` and edge-weighted + multi-scale-gradient losses — tuned to resolve **small",
       "surfaces** (tables, chairs, thin objects) that the v1 wavelet-mean tokenizer averaged away.  It runs",
       "on the **complete DIODE train set** (~87 GB, ~25 k LiDAR views), then inference (a grayscale frame +",
       "optional sparse 3-D context → a **mesh-plane** + **camera pose**).  Weights + logs are saved to",
       f"Google Drive (`{DRIVE_DIR}`).  Budget: ~2.9 M image-embed params, ~16.6 M main (~19.5 M total).",
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

    md("## 3 · Mount Google Drive (persist weights + logs + the dataset archive)",
       "",
       f"Links the local `assets/` (weights) and `renders/` (logs + panels) to `{DRIVE_DIR}`, so **every**",
       "checkpoint is written straight to Drive during training and survives the runtime being recycled.",
       "The compressed DIODE archive is also saved to Drive (next cell), so the ~87 GB download happens",
       "**once** — make sure your Drive has ~90 GB free."),
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

    md("## 4 · Get DIODE: archive on Drive → extract to local (mask-free, balanced indoor/outdoor cap)",
       "",
       "The compressed archive lives on **Drive** (downloaded once, with a progress bar, resumably — only",
       "fetched if missing).  It is then **extracted straight to the local Colab disk** for fast reads.",
       "",
       "Two footprint savers: (1) `*_depth_mask.npy` files are **skipped** — the loader derives the mask",
       "from `depth > 0` (99.97 % identical to the official mask); (2) `PER_CLASS` **caps each class** —",
       "it extracts up to `PER_CLASS` **indoor** *and* `PER_CLASS` **outdoor** views and stops when both",
       "are full, so you get a balanced prototype set (the archive is indoors→outdoor, so it reads through",
       "the indoor block to reach outdoor).  `PER_CLASS = 6000` (~42 GB total) is a good default — set it",
       "anywhere 5000–7000, or `None` for the full ~25 k-view set.  Delete `.train_extracted` to re-extract."),
    code("FULL_TRAIN = True       # True = DIODE train set. False = val split (~2.8 GB).",
         "PER_CLASS = 6000        # extract this many INDOOR + this many OUTDOOR views (None = all)",
         "import os, glob, shutil, tarfile",
         "os.makedirs('data/diode', exist_ok=True)",
         "free_gb = shutil.disk_usage('/content').free / 1e9; print(f'free local disk: {free_gb:.0f} GB')",
         "",
         "def extract_balanced(arc, dest, per_class=None, every=2000):",
         "    # stream the archive (low memory); extract depth + rgb, SKIP masks, cap each class, stop when both full",
         "    n_in = n_out = 0",
         "    with tarfile.open(arc, 'r:gz') as tf:",
         "        for m in tf:",
         "            if m.name.endswith('_depth_mask.npy'):",
         "                continue",
         "            cat = 'in' if 'indoors' in m.name else ('out' if 'outdoor' in m.name else None)",
         "            full = per_class and ((cat == 'in' and n_in >= per_class) or (cat == 'out' and n_out >= per_class))",
         "            if not full:",
         "                try: tf.extract(m, dest, filter='data')",
         "                except TypeError: tf.extract(m, dest)",
         "                if m.name.endswith('_depth.npy'):     # one depth file == one view",
         "                    if cat == 'in': n_in += 1",
         "                    elif cat == 'out': n_out += 1",
         "                    if (n_in + n_out) % every == 0: print(f'  indoor {n_in} | outdoor {n_out}...', flush=True)",
         "            if per_class and n_in >= per_class and n_out >= per_class:",
         "                print(f'  both caps reached (indoor {n_in}, outdoor {n_out}) — stopping.'); break",
         "    return n_in, n_out",
         "",
         "if FULL_TRAIN:",
         "    arc = f'{WS_DRIVE}/data/train.tar.gz' if WS_DRIVE else 'data/train.tar.gz'",
         "    os.makedirs(os.path.dirname(arc), exist_ok=True)",
         "    if not os.path.exists(arc):                       # fetch to Drive only if not already there",
         "        print('downloading DIODE train (~87 GB, resumable, progress) ->', arc)",
         f"        !wget -c -q --show-progress -O \"{{arc}}\" \"{DIODE_TRAIN}\"",
         "    else:",
         "        print('using the DIODE train archive on Drive:', arc, f'({os.path.getsize(arc)/1e9:.0f} GB)')",
         "    flag = 'data/diode/.train_extracted'",
         "    if not os.path.exists(flag):",
         "        if os.path.isdir('data/diode/train'):         # remove any failed/partial extraction first",
         "            print('clearing partial extraction...'); shutil.rmtree('data/diode/train')",
         "        print(f'extracting from Drive -> data/diode (masks skipped, {PER_CLASS} indoor + {PER_CLASS} outdoor)...')",
         "        n_in, n_out = extract_balanced(arc, 'data/diode', per_class=PER_CLASS)",
         "        open(flag, 'w').write(f'{n_in},{n_out}')",
         "        print(f'extracted {n_in} indoor + {n_out} outdoor views')",
         "    else:",
         "        print('already extracted (delete', flag, 'to redo with a different cap)')",
         "else:",
         "    arc = f'{WS_DRIVE}/data/val.tar.gz' if WS_DRIVE else 'data/val.tar.gz'",
         "    os.makedirs(os.path.dirname(arc), exist_ok=True)",
         "    if not glob.glob('data/diode/val/**/*_depth.npy', recursive=True):",
         "        if not os.path.exists(arc):",
         f"            !wget -c -q --show-progress -O \"{{arc}}\" \"{DIODE_VAL}\"",
         "        extract_balanced(arc, 'data/diode')",
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
         "    g, d, m, _ = D.noised_render(scene, Rs[i], ts[i], 512, 256, rng, radius=2)",
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
         "ds = D.FlythroughDataset('auto', img_hw=512, plane_res=256, n_ctx_points=512, max_scene_pts=200000, radius=2)",
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

    md("## 8 · Train the v2 model on the complete train set (A100)",
       "",
       "Learned ConvFPN image embedding (512² input) + Perceiver (`d=320, M=384, L=8`) + multi-level Haar",
       "decoder at `plane_res=256`, on the held-out-by-scene split of the DIODE train set.  `bf16` autocast",
       "and a warmup+cosine LR schedule are on by default.  Checkpoints (`waveletspace_full.pt` / `_latest.pt`)",
       "and render panels are written to `assets/` / `renders/`, which are **linked to Drive**, so they",
       "persist + you can resume.",
       "",
       "This is a multi-hour run.  Reduce `--batch` if you hit OOM (256² output + 512² input is heavier than",
       "v1); resume an interrupted run by adding `--resume assets/waveletspace_full_latest.pt`."),
    code("!python train.py --epochs 30 --batch 8 --img-hw 512 --plane-res 256 --d 320 --M 384 --L 8 --workers 8 --max-scene-pts 200000 --splat-radius 2 --val-cap 256 --out waveletspace_full"),

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
         "    net = WaveletSpaceNet(d=96, M=64, L=3, heads=4, plane_res=64, wave_levels=2, img_size=128); print('untrained model')",
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
