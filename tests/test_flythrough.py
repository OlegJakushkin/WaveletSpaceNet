"""Fly-through data generation: a single fly-through can be generated, rendered and packaged."""
import numpy as np
import torch

from waveletspace import geometry as G, diode as D


def test_flythrough_smooth_valid_and_explores():
    rng = np.random.default_rng(0)
    center = np.array([0, 0, 5.0]); extent = 2.0
    targets = center + rng.normal(0, 1.0, (50, 3))           # parts of the scene to look at
    R, t = G.flythrough(center, extent, rng, n_frames=24, targets=targets)
    assert R.shape == (24, 3, 3) and t.shape == (24, 3)
    # every pose is a proper rotation
    for Ri in R:
        assert np.allclose(Ri @ Ri.T, np.eye(3), atol=1e-5)
        assert abs(np.linalg.det(Ri) - 1.0) < 1e-5
    # smooth: each interpolation step is small vs the overall path span (no teleports)
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    span = np.linalg.norm(t.max(0) - t.min(0)) + 1e-6
    assert steps.max() < span
    # explores DISTANCE: dollies closer / pulls back -> camera-to-centre distance varies
    dist = np.linalg.norm(t - center, axis=1)
    assert dist.max() - dist.min() > 0.2 * extent
    # explores ANGLE: the camera view axis turns to look at different parts
    z = R[:, :, 2]; zmean = z.mean(0); zmean /= np.linalg.norm(zmean)
    assert np.arccos((z * zmean).sum(1).clip(-1, 1)).max() > 0.05


def test_unproject_project_roundtrip():
    rng = np.random.default_rng(1)
    K = G.intrinsics(64, 64, 60.0)
    depth = rng.uniform(2.0, 6.0, (64, 64))
    P, idx = G.unproject_depth(depth, K)
    u, v, z = G.project_points(P, K)
    # unproject then project returns the original pixels/depth
    assert np.allclose(u, idx[:, 1], atol=1e-3)
    assert np.allclose(v, idx[:, 0], atol=1e-3)
    assert np.allclose(z, depth[idx[:, 0], idx[:, 1]], atol=1e-3)


def test_single_flythrough_episode_synthetic():
    rng = np.random.default_rng(2)
    scene = D.synthetic_scene(rng)
    ep = D.make_episode(scene, rng, img_hw=128, plane_res=64, n_ctx_points=256)
    assert ep["img"].shape == (1, 128, 128)
    assert ep["ctx"].shape == (256, 3)
    assert ep["depth"].shape == (1, 64, 64) and ep["mask"].shape == (1, 64, 64)
    assert ep["R"].shape == (3, 3) and ep["t"].shape == (3,)
    assert torch.isfinite(ep["img"]).all() and torch.isfinite(ep["depth"]).all()
    assert float(ep["mask"].mean()) > 0.1          # the render is not empty
    assert float((ep["depth"] > 0).float().mean()) > 0.1


def test_context_has_ten_noise_outliers():
    rng = np.random.default_rng(3)
    scene = D.synthetic_scene(rng)
    clean = scene.subsample(512, np.random.default_rng(3))[0]
    ctx = D.sample_context(scene, 512, np.random.default_rng(3), noise_frac=0.10, n_outliers=10)
    assert ctx.shape == (512, 3)
    # noised context differs from the clean draw (positional noise + 10 outliers)
    assert not np.allclose(ctx, clean)


def test_dataset_reseeds_every_epoch():
    ds = D.FlythroughDataset(None, synthetic=True, img_hw=64, plane_res=32, n_ctx_points=128, length=4)
    ds.set_epoch(0); a = ds[0]["img"].clone()
    ds.set_epoch(1); b = ds[0]["img"].clone()
    assert not torch.allclose(a, b)                 # fresh randomisation per epoch


def test_grouped_split_has_no_scene_leakage():
    # many views across 6 scenes / several scans — the split must not share a scene
    views = [f"root/scene_{s:05d}/scan_{s*10+k:05d}/{s:05d}_{k}_{j}.png"
             for s in range(6) for k in range(3) for j in range(12)]
    tr, vl, info = D.grouped_view_split(views, val_frac=0.34, rng=np.random.default_rng(0))
    tr_scenes = {D.scene_key(v) for v in tr}
    vl_scenes = {D.scene_key(v) for v in vl}
    assert tr_scenes and vl_scenes
    assert tr_scenes.isdisjoint(vl_scenes)                 # no scene in both sets
    assert info["n_train_scenes"] + info["n_val_scenes"] == info["n_scenes"] == 6
    assert len(tr) + len(vl) == len(views)


def test_real_diode_if_available():
    root = D.find_diode_root()
    if root is None:
        return                                       # skip silently when DIODE is absent
    views = D.list_views(root)
    assert len(views) > 0
    scene = D.scene_from_view(views[0], max_pts=20000)
    assert scene.P.shape[0] > 100 and scene.gray.shape[0] == scene.P.shape[0]
    ep = D.make_episode(scene, np.random.default_rng(0), img_hw=128, plane_res=64)
    assert float((ep["depth"] > 0).float().mean()) > 0.05
    # the exploring path frames the surface from several viewpoints (≥1 well-populated),
    # and make_episode's frame selection turns that into a usable target every time.
    rng = np.random.default_rng(1)
    targets, _ = scene.subsample(96, rng)
    Rs, ts = G.flythrough(scene.centroid, scene.extent, rng, n_frames=8, targets=targets)
    fills = [G.splat_render(scene.P, scene.gray, scene.K, Rs[i], ts[i], 96, 96, radius=1)[2].mean()
             for i in range(len(Rs))]
    assert max(fills) > 0.1
    for s in range(4):                                    # every episode yields a populated depth target
        ep = D.make_episode(scene, np.random.default_rng(100 + s), img_hw=128, plane_res=64)
        assert float((ep["depth"] > 0).float().mean()) > 0.03
