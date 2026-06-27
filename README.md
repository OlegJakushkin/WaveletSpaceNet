# WaveletSpaceNet — sparse context + a wavelet image pyramid → mesh-plane + camera pose

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlegJakushkin/WaveletSpaceNet/blob/main/notebooks/waveletspace_colab.ipynb)

WaveletSpaceNet carries the *[Points-as-(Super)Tori](https://github.com/OlegJakushkin) /
WaveletSurfaceNet* idea from **surfaces** to **scenes**.  The surface model reads a
`[context | SEP | main]` token sequence (a sparse summary of the whole shape, a learned
separator, the dense region) and a position-conditioned decoder *emits the Haar wavelet
coefficients* of a distance field.  WaveletSpaceNet keeps that exact skeleton and swaps the
modalities:

```
INPUT   [ sparse 3-D context points | SEP | wavelet image-pyramid tokens ]
            (the cloud gathered            (the current grayscale frame, as a pyramid of
             before — may be EMPTY)         wavelet block embeddings: 1024² → 512² → 256²
                                            → 128² → 64² → 32²)
ENCODER  M Perceiver latents cross-attend the sequence, then L self-attention blocks
OUTPUT  (1) camera pose  — 6-D rotation + translation, RELATIVE to the sparse context
        (2) mesh-plane   — per-pixel depth emitted as 2-D Haar coefficients, inverted to a
                           depth map and unprojected to a grid mesh in the context frame
```

* **Context can be empty.**  With no prior points the sequence is just `[SEP | image]` and
  the pose is predicted in the frame the network anchors to; training randomly drops the
  context so both regimes are learned.
* **Wavelet everywhere.**  The image is tokenised by its 2-D Haar coefficients per block at
  every pyramid level, and the depth head *emits* 2-D Haar coefficients (inverted exactly by
  the orthonormal synthesis filters) — the same multi-scale representation on both ends.
* **Identity start.**  Zero-initialised coefficient heads give a flat plane at the mean
  scene depth, and a bias-initialised pose head starts at `R = I, t = 0` (the source
  viewpoint), so training only learns the *correction*.

The encoder/decoder blocks (`waveletspace/blocks.py`) are the same Perceiver units used by
the surface model's `PerceiverWaveNet`.

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

`notebooks/waveletspace_colab.ipynb` clones this repo, installs, downloads the DIODE
validation split, runs the tests, visualises a generated fly-through, trains on the Colab
GPU and runs inference.  Open it in Colab (after this repo is pushed to
`https://github.com/OlegJakushkin/WaveletSpaceNet`).

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

**Losses** (mirroring the precursor monocular-scene model + the two new heads): log-depth
L1 · normal-from-depth · wavelet-coefficient · camera translation · camera rotation.
**Model selection / eval**: held-out `chamfer(m)` and `logdepthL1`.

---

## Repository layout

| path | what |
|------|------|
| `waveletspace/blocks.py`        | shared Perceiver blocks + Fourier encoding + farthest-point sampling (same units as the surface model). |
| `waveletspace/wavelet2d.py`     | 2-D Haar transform + the `WaveletPyramidTokenizer` (image → multi-scale wavelet block tokens). |
| `waveletspace/geometry.py`      | pinhole camera, 6-D rotations, (un)projection, the splat renderer, and smooth-spline fly-throughs. |
| `waveletspace/model.py`         | `WaveletSpaceNet` — the encoder + pose head + position-conditioned mesh-plane decoder. |
| `waveletspace/diode.py`         | DIODE loader, scene-from-view, fly-through episode generation, the per-epoch-randomised `FlythroughDataset`. |
| `waveletspace/losses.py`        | depth / normal / wavelet / pose losses + chamfer(m) eval. |
| `waveletspace/infer_helpers.py` | mesh-plane (grid mesh) construction + `.obj` export. |
| `train.py` · `infer.py`         | training loop · inference CLI (image + optional context → mesh-plane + pose). |
| `tests/`                        | fly-through generation, model, and local-trainability tests. |
| `notebooks/`                    | the Colab notebook + its builder. |

## License

See [`LICENSE`](LICENSE).
