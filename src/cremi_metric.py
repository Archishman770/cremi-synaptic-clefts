"""CREMI synaptic cleft metric, ported from cremi_python (cremi/evaluation/Clefts.py).

Background in CREMI cleft volumes is 0xffffffffffffffff. Distances are in nm,
computed with an anisotropic EDT at the volume resolution (z, y, x).

  ADGT (FPD): mean distance from each predicted cleft voxel to nearest GT cleft voxel
  ADF  (FND): mean distance from each GT cleft voxel to nearest predicted cleft voxel
  CREMI score = (ADGT + ADF) / 2      (lower is better)
"""
import numpy as np
from scipy import ndimage

BG = np.uint64(0xFFFFFFFFFFFFFFFF)
RES = (40.0, 4.0, 4.0)


def cleft_scores(pred_fg, gt_fg, resolution=RES, threshold=200.0):
    """pred_fg, gt_fg: boolean arrays (True = cleft voxel)."""
    pred_fg = np.asarray(pred_fg, bool)
    gt_fg = np.asarray(gt_fg, bool)
    gt_edt = ndimage.distance_transform_edt(~gt_fg, sampling=resolution)
    pred_edt = ndimage.distance_transform_edt(~pred_fg, sampling=resolution)

    fp_d = gt_edt[pred_fg]
    fn_d = pred_edt[gt_fg]
    adgt = float(fp_d.mean()) if fp_d.size else 0.0
    adf = float(fn_d.mean()) if fn_d.size else 0.0
    return {
        "ADGT": adgt,
        "ADF": adf,
        "CREMI": 0.5 * (adgt + adf),
        "FP_count": int((fp_d > threshold).sum()),
        "FN_count": int((fn_d > threshold).sum()),
        "pred_voxels": int(pred_fg.sum()),
        "gt_voxels": int(gt_fg.sum()),
    }
