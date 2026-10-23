"""Validate on held-out slices (z >= VAL_Z) with the CREMI cleft metric, tune post-processing,
and optionally write CREMI-format submission files for the test volumes A+, B+, C+."""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
from scipy import ndimage

from common import (ROOT, TEST_FILES, TRAIN_FILES, VAL_Z, UNet3D, device, load_clefts, load_raw,
                    postprocess, predict_volume)
from cremi_metric import BG, RES


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = UNet3D(ck["features"])
    m.load_state_dict(ck["model"])
    return m.to(device())


def scores_with_gt_edt(pred_fg, gt_fg, gt_edt):
    fp_d = gt_edt[pred_fg]
    fn_d = ndimage.distance_transform_edt(~pred_fg, sampling=RES)[gt_fg] if pred_fg.any() else None
    adgt = float(fp_d.mean()) if fp_d.size else 0.0
    adf = float(fn_d.mean()) if fn_d is not None else float("inf")
    return {"ADGT": adgt, "ADF": adf, "CREMI": 0.5 * (adgt + adf)}


def validate(model, out_dir, cached=False):
    ctx = 8  # extra slices of z context above the validation block
    preds, gts, edts = {}, {}, {}
    for name, path in TRAIN_FILES.items():
        t0 = time.time()
        cache = os.path.join(out_dir, f"val_pred_{name}.npy")
        if cached:
            p = np.load(cache).astype(np.float32)
        else:
            p = predict_volume(model, load_raw(path)[VAL_Z - ctx:])[ctx:]
            np.save(cache, p.astype(np.float16))
        preds[name] = p
        gts[name] = load_clefts(path)[VAL_Z:]
        edts[name] = ndimage.distance_transform_edt(~gts[name], sampling=RES)
        print(f"predicted val {name} {p.shape} in {time.time() - t0:.0f}s", flush=True)

    results = []
    grid = [(th, sd, ms) for th in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            for sd in [None, 0.3, 0.5, 0.7] for ms in [0, 500] if sd is None or sd > th]
    for th, sd, ms in grid:
            per = {n: scores_with_gt_edt(postprocess(preds[n], th, ms, sd), gts[n], edts[n]) for n in preds}
            avg = float(np.mean([s["CREMI"] for s in per.values()]))
            results.append({"thresh": th, "seed": sd, "min_size": ms, "avg_CREMI": avg, "per_sample": per})
            print(f"th {th:+.1f} seed {sd} min {ms:5d}  CREMI {avg:7.2f}  " +
                  "  ".join(f"{n}: {s['CREMI']:.1f} (ADGT {s['ADGT']:.1f}, ADF {s['ADF']:.1f})"
                            for n, s in per.items()), flush=True)
    best = min(results, key=lambda r: r["avg_CREMI"])
    print("BEST", json.dumps(best, indent=1))
    json.dump({"grid": results, "best": best}, open(os.path.join(out_dir, "val_results.json"), "w"), indent=1)
    return best


def write_submission(model, thresh, min_size, seed, out_dir):
    for name, path in TEST_FILES.items():
        t0 = time.time()
        raw = load_raw(path)
        pred = predict_volume(model, raw)
        np.save(os.path.join(out_dir, f"test_pred_{name}.npy"), pred.astype(np.float16))
        fg = postprocess(pred, thresh, min_size, seed)
        lab, n = ndimage.label(fg)
        clefts = np.full(fg.shape, BG, dtype=np.uint64)
        clefts[fg] = lab[fg].astype(np.uint64)
        out = os.path.join(out_dir, f"sample_{name}_clefts_prediction.hdf")
        with h5py.File(out, "w") as f:
            ds = f.create_dataset("volumes/labels/clefts", data=clefts, compression="gzip")
            ds.attrs["resolution"] = np.array(RES)
        print(f"wrote {out}: {n} clefts, {fg.mean():.4%} fg, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "runs", "unet", "model.pt"))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--submit-model", default=None, help="model for test predictions (default: --model)")
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--min-size", type=int, default=0)
    ap.add_argument("--seed", type=float, default=None)
    ap.add_argument("--cached", action="store_true", help="re-score cached validation predictions")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.model)
    model = load_model(args.model)
    if args.thresh is None:
        best = validate(model, out_dir, args.cached)
        th, ms, sd = best["thresh"], best["min_size"], best["seed"]
    else:
        th, ms, sd = args.thresh, args.min_size, args.seed
    if args.submit:
        sub = os.path.join(ROOT, "submission")
        os.makedirs(sub, exist_ok=True)
        m = load_model(args.submit_model) if args.submit_model else model
        write_submission(m, th, ms, sd, sub)


if __name__ == "__main__":
    main()
