# WaveletSpaceNet v2 — sparse context + a learned image embedding → mesh-plane + camera pose

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlegJakushkin/WaveletSpaceNet/blob/main/notebooks/waveletspace_colab.ipynb)

WaveletSpaceNet carries the *[Points-as-(Super)Tori](https://github.com/OlegJakushkin) /
WaveletSurfaceNet* idea from **surfaces** to **scenes**.  The surface model reads a
`[context | SEP | main]` token sequence (a sparse summary of the whole shape, a learned
separator, the dense region) and a position-conditioned decoder *emits the Haar wavelet
coefficients* of a distance field.  WaveletSpaceNet keeps that skeleton and swaps the
modalities:

```
INPUT   [ sparse 3-D context points | SEP | learned image tokens ]
            (the cloud gathered            (the current grayscale frame -> [gray | sobel |
             before — may be EMPTY)         local-variance] -> ConvNeXt-lite + FPN tokens)
ENCODER  M Perceiver latents cross-attend the sequence, then L self-attention blocks
OUTPUT  (1) camera pose  — 6-D rotation + translation, RELATIVE to the sparse context
        (2) mesh-plane   — per-pixel depth emitted as a MULTI-LEVEL Haar pyramid (LL0 + J
                           detail octaves), inverted to a depth map and unprojected to a
                           grid mesh in the context frame
```

**v2 redesign — to resolve small surfaces (tables, chairs, thin objects).**  v1 tokenised
the image by *average-pooling* signed Haar detail bands, which annihilates the high
frequencies that *are* a chair leg, and emitted a single Haar octave from global latents.
v2 fixes the four stages where small-surface information was lost:

* **Learned image embedding** (`waveletspace/encoder.py`, ≤10 M budget).  A ConvNeXt-lite +
  FPN replaces the Haar-mean tokenizer; the input is grayscale + *computed* edge channels
  (sobel / local-variance), so the path stays sensor-agnostic (visible / IR / night /
  underwater).  The finest stride-4 feature map is handed to the decoder's local gather.
* **Local → coarse routing.**  The coarse (LL) depth band now reads *local* image evidence
  (`ll_local_head`), not only the global latents, so a chair's coarse depth can come from
  the chair's own pixels instead of the scene average.
* **Multi-level wavelet decode.**  The decoder emits LL0 + `J` detail octaves at
  `plane_res=256`, so a thin object gets coefficients at several scales (deep detail path:
  LocalAttn → conv-mix → LocalAttn over a window of the finest feature map).
* **Losses that reward small surfaces.**  Edge-weighted depth L1 + multi-scale gradient
  matching + detail-up-weighted multi-level wavelet, and a **bidirectional** chamfer for
  model selection (so *failing to cover* a thin object is finally penalised).

* **Context can be empty.**  With no prior points the sequence is just `[SEP | image]` and
  the pose is predicted in the frame the network anchors to; training randomly drops the
  context so both regimes are learned.
* **Identity start.**  Zero-initialised coefficient heads give a flat plane at the mean
  scene depth, and a bias-initialised pose head starts at `R = I, t = 0` (the source
  viewpoint), so training only learns the *correction*.

The Perceiver encoder/decoder blocks (`waveletspace/blocks.py`) are the same units used by
the surface model's `PerceiverWaveNet`; the depth field is still *emitted* in the Haar
wavelet domain (`waveletspace/wavelet2d.py`).

---

## Quickstart

### Docker + GPU (no host setup)

```bash
docker compose build                                          # one-time
docker compose run --rm test                                  # the test suite
docker compose run --rm train --epochs 40 --batch 8           # train on DIODE fly-throughs
docker compose run --rm infer --demo --out out/plane.obj      # one frame -> mesh-plane + pose
```

### Bare host (CPU works for the smoke path)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # a single fly-through generates + trains locally
python train.py --smoke             # tiny end-to-end run (synthetic if DIODE is absent)
python infer.py --demo --device cpu --out out/plane.obj
```

### Google Colab

`notebooks/waveletspace_colab.ipynb` is tuned for a **Colab A100 40 GB** runtime: it clones this
repo, installs, **mounts Google Drive**, **streams the complete DIODE train set (~87 GB)**, runs the
tests, visualises a generated fly-through, trains the **complete model** (full `1024→32` pyramid,
`plane_res=128`) and runs inference.  Trained weights (`assets/`) and logs (`renders/`) are linked to
Drive, so **every checkpoint is saved to `MyDrive/WaveletSpaceNet/` during training** (resumable) and a
snapshot of the generated training data is written there too.  Set `FULL_TRAIN = False` in the data
cell to use the small 2.8 GB validation split instead.  Click the badge above (or open it in Colab).

---

## Training data — randomised DIODE fly-throughs

Training uses **DIODE** (Vasiljevic et al.): real dense LiDAR depth, **325 indoor + 446
outdoor** validation scans (`data/diode/val/...`).  Each scene is turned into a *self-consistent
fly-through episode*, re-randomised every epoch:

1. **Scene** — one DIODE view's valid depth is unprojected into a coloured 3-D point cloud;
   the source camera frame is the **context frame** (origin at the source eye).
2. **Fly-through** — a randomised smooth **Catmull-Rom** camera curve that keeps the cloud in
   view (`waveletspace.geometry.flythrough`).
3. **Noised render** — the cloud is splatted into a fly-through camera to produce a *noised
   grayscale frame* plus ground-truth depth/mask (the mesh-plane target).
4. **Sparse context** — a sparse, *noised* subsample of the cloud (positional noise at the
   10 % level **and** ten spurious outlier points — "10 noise in the context").

Because unprojection and re-projection use the *same* intrinsics, the episode is fully
self-consistent and needs no external calibration.  When DIODE is not present a synthetic
scene generator keeps the tests and notebook runnable.

**Losses** (v2, tuned to reward small surfaces): **edge-weighted** log-depth L1 ·
**multi-scale gradient** matching · **multi-level** wavelet-coefficient (detail bands
up-weighted) · low-weight normal · camera translation · camera rotation.
**Model selection / eval**: held-out (split by whole scene, no leakage) **bidirectional**
`chamfer(m)` and `logdepthL1`.

The v2 run used in the notebook (learned ConvFPN embedding + multi-level Haar decode, A100):

```bash
python train.py --epochs 30 --batch 8 --img-hw 512 --plane-res 256 \
    --d 320 --M 384 --L 8 --workers 8 \
    --max-scene-pts 200000 --splat-radius 2 --val-cap 256 --out waveletspace_full
```

`bf16` autocast and a warmup+cosine LR are on by default; `--max-scene-pts` (cloud density,
edge-biased) keeps thin structures in the GT; `--splat-radius` fills the input render (GT is
resolution-aware); resume with `--resume assets/waveletspace_full_latest.pt`.  The model is
~19.5 M params (≈2.9 M image-embed, ≈16.6 M main) of the 30 M+10 M budget.

---

## Repository layout

| path | what |
|------|------|
| `waveletspace/blocks.py`        | shared Perceiver blocks + Fourier encoding + farthest-point sampling (same units as the surface model). |
| `waveletspace/encoder.py`       | the learned image embedding: grayscale→edge adapter + ConvNeXt-lite + FPN tokenizer + windowed neighbour gather. |
| `waveletspace/wavelet2d.py`     | 2-D Haar transform + multi-level `haar_analysis` / `haar_synthesis` for the decoder + wavelet loss. |
| `waveletspace/geometry.py`      | pinhole camera, 6-D rotations, (un)projection, the splat renderer, and smooth-spline fly-throughs. |
| `waveletspace/model.py`         | `WaveletSpaceNet` — learned-embedding encoder + pose head + multi-level wavelet mesh-plane decoder. |
| `waveletspace/diode.py`         | DIODE loader, scene-from-view, fly-through episode generation, the per-epoch-randomised `FlythroughDataset`. |
| `waveletspace/losses.py`        | depth / normal / wavelet / pose losses + chamfer(m) eval. |
| `waveletspace/infer_helpers.py` | mesh-plane (grid mesh) construction + `.obj` export. |
| `train.py` · `infer.py`         | training loop · inference CLI (image + optional context → mesh-plane + pose). |
| `tests/`                        | fly-through generation, model, and local-trainability tests. |
| `notebooks/`                    | the Colab notebook + its builder. |

## License

See [`LICENSE`](LICENSE).
