"""Local trainability: a few steps on synthetic fly-throughs reduce the loss (CPU)."""
import torch

from waveletspace import diode as D, losses as L
from waveletspace.model import WaveletSpaceNet, save_checkpoint, load_checkpoint


def test_train_a_few_steps_reduces_loss():
    torch.manual_seed(0)
    ds = D.FlythroughDataset(None, synthetic=True, img_hw=64, plane_res=32,
                             n_ctx_points=128, length=8)
    ld = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, collate_fn=D.collate)
    net = WaveletSpaceNet(d=64, M=32, L=2, heads=4, plane_res=32, wave_levels=2, win=3,
                          n_ctx=16, img_size=64, enc_widths=(24, 32, 48, 64),
                          enc_depths=(1, 1, 1, 1), fpn_dim=32)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)

    first, last, n = None, None, 0
    for ep in range(6):
        ds.set_epoch(ep)
        for batch in ld:
            out = net(batch["img"], batch["ctx"])
            loss, parts = L.space_loss(out, batch, net)
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if first is None:
                first = parts["loss"]
            last = parts["loss"]; n += 1
    assert n >= 6
    assert last < first                      # the model is learning


def test_checkpoint_save_load(tmp_path):
    net = WaveletSpaceNet(d=64, M=32, L=2, heads=4, plane_res=32, wave_levels=2, win=3,
                          n_ctx=16, img_size=64, enc_widths=(24, 32, 48, 64),
                          enc_depths=(1, 1, 1, 1), fpn_dim=32)
    p = str(tmp_path / "ws.pt")
    save_checkpoint(net, p, epoch=3, val={"chamfer": 1.23})
    net2, ck = load_checkpoint(p)
    assert ck["epoch"] == 3 and ck["val"]["chamfer"] == 1.23
    img = torch.rand(1, 1, 64, 64)
    assert torch.allclose(net(img, None)["depth"], net2(img, None)["depth"], atol=1e-5)
