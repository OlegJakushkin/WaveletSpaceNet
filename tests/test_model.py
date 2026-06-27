"""Model: shapes, identity start, empty context, wavelet invertibility, checkpoint roundtrip."""
import numpy as np
import torch

from waveletspace.model import WaveletSpaceNet, save_checkpoint, load_checkpoint
from waveletspace.wavelet2d import dwt2d, idwt2d
from waveletspace import geometry as G
from waveletspace.infer_helpers import mesh_plane_verts


def _net():
    torch.manual_seed(0)
    return WaveletSpaceNet(d=64, M=32, L=2, heads=4, levels=(128, 64, 32), plane_res=32, k=6, n_ctx=16)


def test_haar2d_invertible():
    x = torch.randn(2, 1, 16, 16)
    assert torch.allclose(idwt2d(dwt2d(x)), x, atol=1e-5)


def test_forward_shapes_and_rotation():
    net = _net()
    img = torch.rand(2, 1, 200, 260)        # non-square, non-pow2 input is handled
    ctx = torch.rand(2, 100, 3) * 4 - 2
    out = net(img, ctx)
    assert out["depth"].shape == (2, 1, 32, 32)
    assert out["coeffs"].shape == (2, 4, 16, 16)
    assert out["R"].shape == (2, 3, 3) and out["t"].shape == (2, 3)
    R = out["R"]
    assert torch.allclose(torch.bmm(R, R.transpose(1, 2)), torch.eye(3)[None].expand(2, -1, -1), atol=1e-4)
    assert torch.allclose(torch.linalg.det(R), torch.ones(2), atol=1e-4)


def test_identity_start_is_flat_mean_depth_plane():
    net = _net()                            # zero-init heads -> flat plane at exp(mean), R=I, t=0
    out = net(torch.rand(1, 1, 64, 64), None)
    d = out["depth"]
    assert d.std() < 1e-3                    # flat
    assert abs(float(d.mean()) - float(torch.exp(net.log_mean))) < 1e-2
    assert torch.allclose(out["R"][0], torch.eye(3), atol=1e-4)
    assert torch.allclose(out["t"][0], torch.zeros(3), atol=1e-4)


def test_empty_context_path():
    net = _net()
    out0 = net(torch.rand(2, 1, 64, 64), None)
    out1 = net(torch.rand(2, 1, 64, 64), torch.zeros(2, 0, 3))   # zero-length context
    assert out0["depth"].shape == out1["depth"].shape == (2, 1, 32, 32)


def test_gradients_flow():
    net = _net()
    out = net(torch.rand(2, 1, 64, 64), torch.rand(2, 50, 3))
    (out["depth"].mean() + out["t"].abs().mean() + out["R"].sum()).backward()
    g = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
    assert torch.isfinite(g) and float(g) > 0


def test_mesh_plane_unprojection():
    net = _net()
    out = net(torch.rand(1, 1, 64, 64), None)
    verts, faces = mesh_plane_verts(out, vfov=60.0)
    assert verts.shape == (32 * 32, 3) and faces.shape[1] == 3
    assert np.isfinite(verts).all()


def test_checkpoint_roundtrip(tmp_path):
    net = _net()
    p = str(tmp_path / "ck.pt")
    save_checkpoint(net, p, epoch=1)
    net2, ck = load_checkpoint(p)
    assert ck["epoch"] == 1
    img = torch.rand(1, 1, 64, 64)
    a = net(img, None)["depth"]; b = net2(img, None)["depth"]
    assert torch.allclose(a, b, atol=1e-5)
