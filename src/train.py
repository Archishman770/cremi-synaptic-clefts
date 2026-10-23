"""Train an anisotropic 3D U-Net to regress tanh-scaled signed distance to synaptic clefts."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from common import (ROOT, TRAIN_FILES, VAL_Z, UNet3D, device, load_clefts, load_raw,
                    sted_target)


def cached_target(name, path):
    cache = os.path.join(ROOT, "runs", "cache", f"sted_{name}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    t = sted_target(load_clefts(path))
    np.save(cache, t)
    return t


class Sampler:
    def __init__(self, raws, targets, zrange, patch, rng, pos_frac=0.5):
        self.raws, self.targets, self.patch, self.rng, self.pos_frac = raws, targets, patch, rng, pos_frac
        self.z0, self.z1 = zrange
        self.pos = []
        for t in targets:
            zz, yy, xx = np.nonzero(t[self.z0:self.z1] > 0)
            idx = rng.choice(len(zz), size=min(len(zz), 200_000), replace=False)
            self.pos.append(np.stack([zz[idx] + self.z0, yy[idx], xx[idx]], 1))

    def _corner(self, i):
        pz, py, px = self.patch
        Z, Y, X = self.raws[i].shape
        if self.rng.random() < self.pos_frac and len(self.pos[i]):
            c = self.pos[i][self.rng.integers(len(self.pos[i]))]
            jitter = self.rng.integers(-np.array(self.patch) // 3, np.array(self.patch) // 3 + 1)
            c = c + jitter - np.array(self.patch) // 2
        else:
            c = np.array([self.rng.integers(self.z0, self.z1 - pz + 1),
                          self.rng.integers(0, Y - py + 1), self.rng.integers(0, X - px + 1)])
        c[0] = np.clip(c[0], self.z0, self.z1 - pz)
        c[1] = np.clip(c[1], 0, Y - py)
        c[2] = np.clip(c[2], 0, X - px)
        return c

    def sample(self, augment=True):
        i = self.rng.integers(len(self.raws))
        z, y, x = self._corner(i)
        pz, py, px = self.patch
        r = self.raws[i][z:z + pz, y:y + py, x:x + px].astype(np.float32) / 255.0
        t = self.targets[i][z:z + pz, y:y + py, x:x + px].astype(np.float32)
        if augment:
            r, t = augment_pair(r, t, self.rng)
        return r, t


def augment_pair(r, t, rng):
    for ax in range(3):
        if rng.random() < 0.5:
            r, t = np.flip(r, ax), np.flip(t, ax)
    if rng.random() < 0.5:
        r, t = r.transpose(0, 2, 1), t.transpose(0, 2, 1)
    # intensity jitter (per-section, mimics ssTEM section contrast variation)
    a = rng.uniform(0.85, 1.15, size=(r.shape[0], 1, 1)).astype(np.float32)
    b = rng.uniform(-0.08, 0.08, size=(r.shape[0], 1, 1)).astype(np.float32)
    r = np.clip(r * a + b, 0, 1)
    r = r ** np.float32(rng.uniform(0.8, 1.25))
    # missing / defect sections
    if rng.random() < 0.15:
        zi = rng.integers(r.shape[0])
        r = r.copy()
        r[zi] = 0.0 if rng.random() < 0.5 else r[zi].mean()
    r = r + rng.normal(0, 0.02, r.shape).astype(np.float32)
    return np.ascontiguousarray(r, np.float32), np.ascontiguousarray(t)


def batch(sampler, bs, augment, dev):
    rs, ts = zip(*(sampler.sample(augment) for _ in range(bs)))
    r = torch.from_numpy(np.stack(rs))[:, None].to(dev)
    t = torch.from_numpy(np.stack(ts))[:, None].to(dev)
    return r, t


def loss_fn(pred, target, crop, wpos_max=5.0):
    cz, cy, cx = crop
    p = pred[..., cz:pred.shape[2] - cz, cy:pred.shape[3] - cy, cx:pred.shape[4] - cx]
    t = target[..., cz:target.shape[2] - cz, cy:target.shape[3] - cy, cx:target.shape[4] - cx]
    pos = (t > 0).float()
    n_pos = pos.sum().clamp(min=1.0)
    w_pos = ((pos.numel() - n_pos) / n_pos).clamp(1.0, wpos_max)
    w = 1.0 + (w_pos - 1.0) * pos
    return (w * (p - t) ** 2).sum() / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--patch", type=int, nargs=3, default=[16, 192, 192])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "unet"))
    ap.add_argument("--full", action="store_true", help="train on all 125 slices (final model)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--pos-frac", type=float, default=0.5)
    ap.add_argument("--wpos-max", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = device()
    rng = np.random.default_rng(1 if args.init else 0)
    torch.manual_seed(0)

    t0 = time.time()
    raws, targets = [], []
    for name, path in TRAIN_FILES.items():
        raws.append(load_raw(path))
        targets.append(cached_target(name, path))
        print(f"loaded {name} {raws[-1].shape} pos={float((targets[-1] > 0).mean()):.4%} "
              f"({time.time() - t0:.0f}s)", flush=True)

    ztrain = (0, raws[0].shape[0]) if args.full else (0, VAL_Z)
    train_s = Sampler(raws, targets, ztrain, tuple(args.patch), rng, args.pos_frac)
    val_s = Sampler(raws, targets, (VAL_Z, raws[0].shape[0]), (16, 192, 192), np.random.default_rng(1))
    val_batches = [batch(val_s, args.bs, False, "cpu") for _ in range(24)]

    model = UNet3D(args.features)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    model = model.to(dev)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))
    crop = (2, 16, 16)

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    t0 = time.time()
    run = 0.0
    for it in range(1, args.iters + 1):
        model.train()
        r, t = batch(train_s, args.bs, True, dev)
        loss = loss_fn(model(r), t, crop, args.wpos_max)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if args.bench and it == 30:
            print(f"bench: {(time.time() - t0) / 30:.3f}s/iter", flush=True)
            return
        if it % 100 == 0:
            print(f"it {it} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / it:.3f}s/it", flush=True)
        if it % 1000 == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_fn(model(r.to(dev)), t.to(dev), crop).item() for r, t in val_batches]))
            print(f"== it {it} val_loss {vl:.4f}", flush=True)
            log.write(json.dumps({"it": it, "train_loss": run, "val_loss": vl, "t": time.time() - t0}) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "features": args.features, "it": it},
                       os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
