"""Shared data loading, model and sliding-window inference for CREMI cleft detection."""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from cremi_metric import BG, RES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TRAIN_FILES = {s: os.path.join(DATA, f"sample_{s}_20160501.hdf") for s in "ABC"}
TEST_FILES = {s + "+": os.path.join(DATA, f"sample_{s}+_20160601.hdf") for s in "ABC"}
VAL_Z = 100  # slices [0, VAL_Z) train, [VAL_Z, 125) held-out validation
STED_SCALE = 50.0  # nm, target = tanh(signed_distance / STED_SCALE)


def load_raw(path):
    with h5py.File(path, "r") as f:
        return f["volumes/raw"][:]


def load_clefts(path):
    with h5py.File(path, "r") as f:
        c = f["volumes/labels/clefts"][:]
    return c != BG


def sted_target(fg, resolution=RES):
    """tanh-scaled signed distance: >0 inside cleft, <0 outside."""
    out_d = ndimage.distance_transform_edt(~fg, sampling=resolution)
    in_d = ndimage.distance_transform_edt(fg, sampling=resolution)
    sd = np.where(fg, in_d, -out_d)
    return np.tanh(sd / STED_SCALE).astype(np.float16)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- model
# Conv3d is slow on MPS, so the network keeps tensors as (B, Z, C, H, W) and expresses every
# convolution as a Conv2d over (B*Z, C, H, W). A 3x3x3 conv is computed exactly by stacking the
# z-1, z, z+1 slices along channels before a 3x3 Conv2d.
class Conv(nn.Module):
    def __init__(self, cin, cout, kz):
        super().__init__()
        self.kz = kz
        self.conv = nn.Conv2d(cin * kz, cout, 3, padding=1)

    def forward(self, x):
        b, z, c, h, w = x.shape
        if self.kz > 1:
            r = self.kz // 2
            zeros = x.new_zeros(b, r, c, h, w)
            xp = torch.cat([zeros, x, zeros], 1)
            x = torch.cat([xp[:, i:i + z] for i in range(self.kz)], 2)
        y = self.conv(x.reshape(b * z, -1, h, w))
        return y.reshape(b, z, -1, h, w)


class Block(nn.Module):
    def __init__(self, cin, cout, kz):
        super().__init__()
        self.c1, self.c2 = Conv(cin, cout, kz), Conv(cout, cout, kz)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)

    def _norm(self, n, x):
        b, z, c, h, w = x.shape
        return F.relu(n(x.reshape(b * z, c, h, w))).reshape(b, z, c, h, w)

    def forward(self, x):
        return self._norm(self.n2, self.c2(self._norm(self.n1, self.c1(x))))


class Down(nn.Module):
    def __init__(self, pz):
        super().__init__()
        self.pz = pz

    def forward(self, x):
        b, z, c, h, w = x.shape
        x = F.max_pool2d(x.reshape(b * z, c, h, w), 2).reshape(b, z, c, h // 2, w // 2)
        if self.pz == 2:
            x = x.reshape(b, z // 2, 2, c, h // 2, w // 2).amax(2)
        return x


class Up(nn.Module):
    def __init__(self, cin, cout, pz):
        super().__init__()
        self.pz = pz
        self.conv = nn.ConvTranspose2d(cin, cout, 2, stride=2)

    def forward(self, x):
        b, z, c, h, w = x.shape
        y = self.conv(x.reshape(b * z, c, h, w)).reshape(b, z, -1, 2 * h, 2 * w)
        if self.pz == 2:
            y = y.repeat_interleave(2, dim=1)
        return y


class UNet3D(nn.Module):
    """Anisotropic U-Net (40x4x4 nm data): pools (1,2,2), (1,2,2), (2,2,2).
    First level uses xy-only kernels; deeper levels use full 3x3x3 kernels.
    Input/output layout is the usual (B, 1, Z, H, W)."""

    def __init__(self, f=16):
        super().__init__()
        ch = [f, f * 2, f * 4, f * 8]
        pz = [1, 1, 2]
        kz = [1, 3, 3, 3]
        self.enc = nn.ModuleList([Block(1 if i == 0 else ch[i - 1], ch[i], kz[i]) for i in range(4)])
        self.down = nn.ModuleList([Down(p) for p in pz])
        self.up = nn.ModuleList([Up(ch[i], ch[i - 1], pz[i - 1]) for i in range(3, 0, -1)])
        self.dec = nn.ModuleList([Block(ch[i - 1] * 2, ch[i - 1], kz[i - 1]) for i in range(3, 0, -1)])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, Z, 1, H, W)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < 3:
                skips.append(x)
                x = self.down[i](x)
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 2))
        b, z, c, h, w = x.shape
        y = torch.tanh(self.head(x.reshape(b * z, c, h, w))).reshape(b, z, 1, h, w)
        return y.permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_volume(model, raw, patch=(16, 256, 256), margin=(2, 24, 24), dev=None, tta=True):
    """Sliding-window prediction over a full (z, y, x) uint8 volume. Returns float32 in [-1, 1]."""
    dev = dev or device()
    model.eval()
    pz, py, px = patch
    mz, my, mx = margin
    # reflect-pad so every output voxel gets a centre-cropped prediction
    padded = np.pad(raw, ((mz, mz + pz), (my, my + py), (mx, mx + px)), mode="reflect")
    vol = torch.from_numpy(padded.astype(np.float32) / 255.0)
    Z, Y, X = raw.shape
    out = np.zeros(raw.shape, np.float32)
    sz, sy, sx = pz - 2 * mz, py - 2 * my, px - 2 * mx
    for z0 in range(0, Z, sz):
        for y0 in range(0, Y, sy):
            for x0 in range(0, X, sx):
                inp = vol[z0:z0 + pz, y0:y0 + py, x0:x0 + px][None, None].to(dev)
                if tta:
                    preds = []
                    for flip in ([], [3], [4], [3, 4]):
                        t = inp.flip(flip) if flip else inp
                        p = model(t)
                        preds.append(p.flip(flip) if flip else p)
                    p = torch.stack(preds).mean(0)
                else:
                    p = model(inp)
                p = p[0, 0, mz:pz - mz, my:py - my, mx:px - mx].float().cpu().numpy()
                ez, ey, ex = min(sz, Z - z0), min(sy, Y - y0), min(sx, X - x0)
                out[z0:z0 + ez, y0:y0 + ey, x0:x0 + ex] = p[:ez, :ey, :ex]
    return out


def postprocess(pred, thresh, min_size, seed=None):
    """Threshold, then drop components smaller than min_size voxels and (hysteresis) components
    whose maximum prediction never exceeds `seed`."""
    fg = pred > thresh
    if min_size > 0 or seed is not None:
        lab, n = ndimage.label(fg)
        if n:
            keep = np.ones(n + 1, bool)
            if min_size > 0:
                keep &= np.bincount(lab.ravel(), minlength=n + 1) >= min_size
            if seed is not None:
                peak = ndimage.maximum(pred, lab, index=np.arange(n + 1))
                keep &= np.asarray(peak) > seed
            keep[0] = False
            fg = keep[lab]
    return fg
