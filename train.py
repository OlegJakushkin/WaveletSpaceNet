"""Train WaveletSpaceNet on randomised DIODE fly-throughs.

Each epoch, every scene is re-rendered along a *fresh* random smooth fly-through with
fresh noise and a fresh noised sparse context (the dataset reseeds per epoch).  The
model learns to predict, from a single noised grayscale frame (+ optional sparse
context), the camera pose relative to the context and a mesh-plane (per-pixel depth).
Model selection is by held-out chamfer(m); per-epoch render panels are written to
``renders/``.

  python train.py --smoke                       # tiny CPU/GPU sanity run (synthetic if no DIODE)
  python train.py --epochs 40 --batch 8         # full run (GPU)
  docker compose run --rm train --epochs 40     # in the GPU container
"""
import argparse, json, os, time
import numpy as np, torch

from waveletspace import diode as D, losses as L
from waveletspace.model import WaveletSpaceNet, save_checkpoint


def parse_levels(s):
    return tuple(int(x) for x in str(s).replace(" ", "").split(","))


def render_panel(net, batch, path, dev, n=4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  render skip:", e, flush=True); return
    net.eval()
    with torch.no_grad():
        out = net(batch["img"].to(dev), batch["ctx"].to(dev))
    n = min(n, batch["img"].shape[0])
    fig, ax = plt.subplots(3, n, figsize=(3 * n, 8))
    ax = ax.reshape(3, n)
    for i in range(n):
        ax[0, i].imshow(batch["img"][i, 0].cpu(), cmap="gray"); ax[0, i].set_title(f"in {batch['name'][i][:14]}", fontsize=7)
        gt = batch["depth"][i, 0].cpu().numpy(); gt = np.where(gt > 0, gt, np.nan)
        ax[1, i].imshow(gt, cmap="turbo"); ax[1, i].set_title("GT depth", fontsize=7)
        ax[2, i].imshow(out["depth"][i, 0].cpu().numpy(), cmap="turbo"); ax[2, i].set_title("pred depth", fontsize=7)
    for a in ax.ravel(): a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    net.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--img-hw", type=int, default=256)
    ap.add_argument("--plane-res", type=int, default=64)
    ap.add_argument("--levels", type=str, default="1024,512,256,128,64,32")
    ap.add_argument("--n-ctx-points", type=int, default=512)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--empty-ctx-prob", type=float, default=0.2, help="fraction of steps with NO context")
    ap.add_argument("--ctx-min", type=int, default=8)
    ap.add_argument("--ctx-max", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-cap", type=int, default=256, help="cap #val episodes/epoch (0 = all)")
    ap.add_argument("--cap", type=int, default=0, help="cap #scenes (0 = all)")
    ap.add_argument("--max-scene-pts", type=int, default=60000, help="points kept per scene cloud")
    ap.add_argument("--splat-radius", type=int, default=1, help="render splat radius (raise for dense/hi-res)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--diode-root", type=str, default="auto")
    ap.add_argument("--device", type=str,
                    default="cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    ap.add_argument("--out", type=str, default="waveletspace")
    ap.add_argument("--resume", type=str, default="")
    a = ap.parse_args()

    levels = parse_levels(a.levels)
    if a.smoke:
        a.epochs, a.batch, a.img_hw, levels = 2, 4, 128, (128, 64, 32)
        a.d, a.M, a.L, a.heads, a.cap, a.workers = 96, 64, 3, 4, 24, 0
        a.device = "cpu"
    dev = a.device
    os.makedirs("renders", exist_ok=True); os.makedirs("assets", exist_ok=True)
    t0 = time.time()

    # ---- train / val view split -------------------------------------------
    root = None if a.smoke else (D.find_diode_root() if a.diode_root == "auto" else a.diode_root)
    views = D.list_views(root) if root else []
    if a.smoke or not views:
        print("no DIODE -> synthetic scenes", flush=True)
        tr_views = vl_views = None; synth = True
    else:
        # split by whole SCENE (no scene in both sets) so held-out chamfer is honest
        rng = np.random.default_rng(0)
        tr_views, vl_views, info = D.grouped_view_split(views, a.val_frac, rng, cap=a.cap)
        synth = False
        print(f"DIODE {root}: {info['n_train_views']} train / {info['n_val_views']} val views | "
              f"{info['n_train_scenes']} train / {info['n_val_scenes']} val SCENES "
              f"(of {info['n_scenes']})", flush=True)
        if info["n_scenes"] <= 8:
            print(f"  ! only {info['n_scenes']} distinct scenes — val chamfer is high-variance", flush=True)

    cfg = dict(img_hw=a.img_hw, plane_res=a.plane_res, n_ctx_points=a.n_ctx_points,
               radius=a.splat_radius)
    tr = D.FlythroughDataset(root, views=tr_views, synthetic=synth, seed=1,
                             max_scene_pts=a.max_scene_pts,
                             length=(a.cap or None) if synth else None, **cfg)
    val_len = 16 if synth else (a.val_cap or None)
    vl = D.FlythroughDataset(root, views=vl_views, synthetic=synth, seed=777,
                             max_scene_pts=a.max_scene_pts, length=val_len, **cfg)
    if synth and a.smoke:
        tr._len = a.cap
    ld = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, collate_fn=D.collate, drop_last=True)
    vld = torch.utils.data.DataLoader(vl, batch_size=a.batch, shuffle=False,
                                      num_workers=0, collate_fn=D.collate)

    # ---- model ------------------------------------------------------------
    torch.manual_seed(0)
    net = WaveletSpaceNet(d=a.d, M=a.M, L=a.L, heads=a.heads, levels=levels,
                          plane_res=a.plane_res, n_ctx=a.ctx_max).to(dev)
    print(f"WaveletSpaceNet {net.count_params():,} params | img tokens {net.img_tok.n_tokens()} "
          f"| levels {levels} | device {dev}", flush=True)
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state"]); print(f"resumed from {a.resume}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(0)

    @torch.no_grad()
    def validate():
        net.eval(); agg = {"logdepthL1": 0.0, "chamfer": 0.0}; nb = 0
        for vb in vld:
            out = net(vb["img"].to(dev), vb["ctx"].to(dev))
            m = L.eval_metrics(out, vb, net)
            for k in agg: agg[k] += m[k]
            nb += 1
        net.train(); return {k: v / max(nb, 1) for k, v in agg.items()}

    best, hist = float("inf"), []
    vbatch = next(iter(vld))
    for ep in range(a.epochs):
        tr.set_epoch(ep); run = {}; nb = 0
        for batch in ld:
            # flexible context: random split, and sometimes NO context (empty-context training)
            nctx = int(torch.randint(a.ctx_min, a.ctx_max + 1, (1,), generator=g).item())
            ctx = None if (torch.rand(1, generator=g).item() < a.empty_ctx_prob) else batch["ctx"].to(dev)
            out = net(batch["img"].to(dev), ctx, n_ctx=nctx)
            loss, parts = L.space_loss(out, batch, net)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                for k, v in parts.items(): run[k] = run.get(k, 0.0) + v
                nb += 1
            if nb and nb % 25 == 0:
                gpu = f"| GPU {torch.cuda.max_memory_allocated()/1e9:.1f}GB " if dev == "cuda" else ""
                print(f"  ep{ep+1} step{nb} loss {run['loss']/nb:.4f} depth {run['depth']/nb:.3f} "
                      f"rot {run['rot']/nb:.3f} trans {run['trans']/nb:.3f} {gpu}| {time.time()-t0:.0f}s", flush=True)
        val = validate()
        render_panel(net, vbatch, f"renders/ws_val_ep{ep+1}.png", dev)
        improved = val["chamfer"] < best
        meta = {"epoch": ep + 1, "train": {k: run[k] / max(nb, 1) for k in run}, "val": val}
        hist.append(meta)
        save_checkpoint(net, f"assets/{a.out}_latest.pt", **meta)
        if improved:
            best = val["chamfer"]; save_checkpoint(net, f"assets/{a.out}.pt", **meta)
        print(f"epoch {ep+1}/{a.epochs}: loss {run.get('loss',0)/max(nb,1):.4f} | "
              f"val logdepthL1 {val['logdepthL1']:.4f} chamfer(m) {val['chamfer']:.4f}"
              f"{'  *saved*' if improved else ''} | {time.time()-t0:.0f}s", flush=True)
        json.dump(hist, open("renders/ws_train_hist.json", "w"), indent=1)
    print(f"DONE in {time.time()-t0:.0f}s | best chamfer(m) {best:.4f} | assets/{a.out}.pt", flush=True)


if __name__ == "__main__":
    main()
