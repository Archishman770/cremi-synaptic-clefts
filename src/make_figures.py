"""Make result figures and tables for the README (writes to figs/ and results/)."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, TEST_FILES, load_raw

FIGS = os.path.join(ROOT, "figs")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SAMPLE_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
LEADER_TOP = 54.23  # CREMI cleft leaderboard #1 (hidden test set, avg of A+/B+/C+)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

CHECKPOINTS = [
    ("Stage 1\niter 1k", "runs/ck1000/val_results.json"),
    ("Stage 1\niter 4k", "runs/ck4000/val_results.json"),
    ("Stage 1\niter 8k", "runs/unet/val_results.json"),
    ("Stage 2\nfine-tuned", "runs/ft/val_results.json"),
]


def load_grid(rel):
    return json.load(open(os.path.join(ROOT, rel)))


def best(grid):
    return min(grid["grid"], key=lambda g: g["avg_CREMI"])


def read_log(rel):
    return [json.loads(l) for l in open(os.path.join(ROOT, rel))]


def fig_training():
    s1, s2 = read_log("runs/unet/log.jsonl"), read_log("runs/ft/log.jsonl")
    s1 = [r for r in s1 if r["it"] <= 8000]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    it1 = [r["it"] for r in s1]
    it2 = [8000 + r["it"] for r in s2]
    ax.plot(it1 + it2, [r["val_loss"] for r in s1 + s2], color=SAMPLE_COLORS["B"], lw=2,
            marker="o", ms=6, label="validation loss (held-out slices)")
    ax.plot(it1, [r["train_loss"] for r in s1], color=SAMPLE_COLORS["A"], lw=2, marker="o", ms=6,
            label="training loss, stage 1")
    ax.axvline(8000, color=INK2, lw=1, ls="--")
    ax.text(8100, ax.get_ylim()[1] * 0.97, "stage 2: fine-tune\nagainst false positives",
            va="top", fontsize=9, color=INK2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("weighted L2 loss")
    ax.set_title("Training curves")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_training_curves.png"), dpi=160)
    plt.close(fig)


def fig_progress():
    rows = []
    for label, rel in CHECKPOINTS:
        b = best(load_grid(rel))
        rows.append((label, b))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(rows))
    w = 0.24
    for k, s in enumerate("ABC"):
        vals = [r[1]["per_sample"][s]["CREMI"] for r in rows]
        ax.bar(x + (k - 1) * w, vals, w - 0.03, color=SAMPLE_COLORS[s], label=f"sample {s}", zorder=3)
    for i, (_, b) in enumerate(rows):
        ax.text(i, max(b["per_sample"][s]["CREMI"] for s in "ABC") + 8, f"avg {b['avg_CREMI']:.0f}",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.axhline(LEADER_TOP, color=INK, lw=1.2, ls="--", zorder=4,
               label=f"leaderboard #1: {LEADER_TOP} nm (test set)")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("CREMI score, nm (lower is better)")
    ax.set_title("Validation score by checkpoint")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=10)
    ax.set_ylim(0, 400)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_cremi_by_checkpoint.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(RESULTS, "validation_scores.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["checkpoint", "threshold", "min_size", "seed", "sample", "ADGT_nm", "ADF_nm", "CREMI_nm"])
        for label, b in rows:
            for s in "ABC":
                p = b["per_sample"][s]
                wr.writerow([label.replace("\n", " "), b["thresh"], b["min_size"], b.get("seed"), s,
                             round(p["ADGT"], 2), round(p["ADF"], 2), round(p["CREMI"], 2)])
            wr.writerow([label.replace("\n", " "), b["thresh"], b["min_size"], b.get("seed"), "avg",
                         "", "", round(b["avg_CREMI"], 2)])


def fig_errors():
    b = best(load_grid("runs/ft/val_results.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    adgt = [b["per_sample"][s]["ADGT"] for s in "ABC"]
    adf = [b["per_sample"][s]["ADF"] for s in "ABC"]
    ax.bar(x - 0.2, adgt, 0.37, color=SAMPLE_COLORS["A"], label="ADGT: false-positive distance", zorder=3)
    ax.bar(x + 0.2, adf, 0.37, color=SAMPLE_COLORS["B"], label="ADF: false-negative distance", zorder=3)
    for i in range(3):
        ax.text(i - 0.2, adgt[i] + 5, f"{adgt[i]:.0f}", ha="center", fontsize=9, color=INK2)
        ax.text(i + 0.2, adf[i] + 5, f"{adf[i]:.0f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, [f"sample {s}" for s in "ABC"])
    ax.set_ylabel("mean distance, nm")
    ax.set_title("Where the error comes from (fine-tuned model)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_error_breakdown.png"), dpi=160)
    plt.close(fig)


def fig_threshold():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#c3c2b7", "#8f8d86", SAMPLE_COLORS["A"], SAMPLE_COLORS["B"]]
    with open(os.path.join(RESULTS, "threshold_sweep.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["checkpoint", "threshold", "min_size", "seed", "avg_CREMI_nm", "A", "B", "C"])
        for (label, rel), c in zip(CHECKPOINTS, colors):
            grid = load_grid(rel)["grid"]
            for g in grid:
                wr.writerow([label.replace("\n", " "), g["thresh"], g["min_size"], g.get("seed"),
                             round(g["avg_CREMI"], 2)] + [round(g["per_sample"][s]["CREMI"], 2) for s in "ABC"])
            pts = sorted((g["thresh"], g["avg_CREMI"]) for g in grid
                         if g["min_size"] == 0 and g.get("seed") is None and np.isfinite(g["avg_CREMI"]))
            ax.plot(*zip(*pts), color=c, lw=2, marker="o", ms=6, label=label.replace("\n", " "))
    ax.axhline(LEADER_TOP, color=INK, lw=1.2, ls="--", label=f"leaderboard #1 ({LEADER_TOP} nm, test set)")
    ax.set_xlabel("threshold on predicted tanh(signed distance)")
    ax.set_ylabel("avg CREMI score, nm")
    ax.set_ylim(0, 450)
    ax.set_title("Threshold sweep on validation")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_threshold_sweep.png"), dpi=160)
    plt.close(fig)


def fig_test_overlays(z=62):
    tiles = []
    for name, path in TEST_FILES.items():
        raw = load_raw(path)[z]
        pred = np.load(os.path.join(ROOT, "submission", f"test_pred_{name}.npy"), mmap_mode="r")[z]
        fg = np.asarray(pred, np.float32) > -0.2
        rgb = np.stack([raw] * 3, -1).astype(np.float32)
        rgb[fg] = rgb[fg] * 0.35 + np.array([255, 40, 160]) * 0.65
        tiles.append(Image.fromarray(rgb.astype(np.uint8)).resize((600, 600)))
    canvas = Image.new("RGB", (1800, 600))
    for i, t in enumerate(tiles):
        canvas.paste(t, (600 * i, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=36)
    for i, name in enumerate(TEST_FILES):
        draw.rectangle([600 * i + 10, 10, 600 * i + 90, 60], fill=(255, 255, 255))
        draw.text((600 * i + 20, 14), name, fill=(0, 0, 0), font=font)
    canvas.save(os.path.join(FIGS, "fig6_test_predictions_A+_B+_C+.png"))


if __name__ == "__main__":
    fig_training()
    fig_progress()
    fig_errors()
    fig_threshold()
    fig_test_overlays()
    print("done")
