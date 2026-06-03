"""
Figure generation for the paper.
  Fig 1: Frequency-band error analysis (gap analysis)
  Fig 2: Ablation table bar chart
  Fig 3: PSNR vs #Params scatter (identical budget comparison)
  Fig 4: Lambda_freq sweep
  Fig 5: Training curves
  Fig 6: Visual SR comparison (LR / Bicubic / Baselines / Proposed)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
SAVE_DPI = 150


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Fig 1 — Frequency-band error analysis (gap analysis)
# ─────────────────────────────────────────────────────────────────

def plot_freq_band_analysis(band_results: dict, save_path: str):
    """
    band_results: {model_name: {low, mid, high}, ...}
    Shows that existing models fail at high-frequency reconstruction.
    """
    names  = list(band_results.keys())
    bands  = ["low", "mid", "high"]
    labels = ["Low Freq.", "Mid Freq.", "High Freq."]
    x      = np.arange(len(names))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (band, label) in enumerate(zip(bands, labels)):
        vals = [band_results[n][band] for n in names]
        ax.bar(x + i*width, vals, width, label=label,
               color=COLORS[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean Absolute Error (FFT magnitude)")
    ax.set_title("Frequency-Band Reconstruction Error Analysis\n"
                 "Higher error in high-frequency band reveals the research gap")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────
# Fig 2 — Ablation bar chart (PSNR)
# ─────────────────────────────────────────────────────────────────

def plot_ablation_bars(ablation_results: dict, save_path: str, scale: int = 2):
    names = list(ablation_results.keys())
    psnr  = [ablation_results[n]["PSNR"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, psnr, color=COLORS[:len(names)], edgecolor="white", alpha=0.88)
    for bar, val in zip(bars, psnr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Ablation Study — PSNR on DIV2K Validation (×{scale})")
    ax.set_ylim(min(psnr) - 1, max(psnr) + 1)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────
# Fig 3 — PSNR vs #Params (identical budget comparison)
# ─────────────────────────────────────────────────────────────────

def plot_psnr_vs_params(results: dict, save_path: str, scale: int = 2):
    """
    Scatter plot: X = #params (M), Y = PSNR (dB).
    Shows FA-SwinLite achieves competitive PSNR with fewer params.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, m) in enumerate(results.items()):
        ax.scatter(m["Params_M"], m["PSNR"], s=120, color=COLORS[i % len(COLORS)],
                   zorder=5, label=name)
        ax.annotate(name, (m["Params_M"], m["PSNR"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Model Parameters (M)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"PSNR vs. #Parameters Comparison (×{scale})\n"
                 "All models evaluated on DIV2K-Val")
    ax.grid(linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────
# Fig 4 — Lambda_freq sweep
# ─────────────────────────────────────────────────────────────────

def plot_lambda_sweep(sweep_results: dict, save_path: str, scale: int = 2):
    lam   = [v["lambda_freq"] for v in sweep_results.values()]
    psnr  = [v["PSNR"]        for v in sweep_results.values()]
    ssim  = [v["SSIM"]        for v in sweep_results.values()]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(lam, psnr, "o-", color=COLORS[0], label="PSNR (dB)", linewidth=2)
    ax2.plot(lam, ssim, "s--", color=COLORS[1], label="SSIM",      linewidth=2)
    ax1.set_xlabel("λ_freq (Frequency Loss Weight)")
    ax1.set_ylabel("PSNR (dB)",  color=COLORS[0])
    ax2.set_ylabel("SSIM",       color=COLORS[1])
    ax1.set_title(f"Effect of Frequency Loss Weight λ_freq (×{scale})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
    ax1.grid(linestyle="--", alpha=0.4)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────
# Fig 5 — Training curves
# ─────────────────────────────────────────────────────────────────

def plot_training_curves(log_paths: dict, save_path: str, scale: int = 2):
    """
    log_paths: {model_name: path_to_json_log}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, (name, path) in enumerate(log_paths.items()):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        epochs = [e["epoch"] for e in log]
        losses = [e["loss"]  for e in log]
        psnrs  = [e["psnr"]  for e in log]
        c      = COLORS[i % len(COLORS)]
        axes[0].plot(epochs, losses, color=c, label=name, linewidth=1.5)
        axes[1].plot(epochs, psnrs,  color=c, label=name, linewidth=1.5)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
    axes[0].set_title(f"Training Loss Curves (×{scale})")
    axes[0].legend(fontsize=8); axes[0].grid(linestyle="--", alpha=0.4)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val PSNR (dB)")
    axes[1].set_title(f"Validation PSNR Curves (×{scale})")
    axes[1].legend(fontsize=8); axes[1].grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────
# Fig 6 — Visual comparison grid
# ─────────────────────────────────────────────────────────────────

def plot_visual_comparison(result_dirs: dict, save_path: str, n_images: int = 3):
    """
    result_dirs: {model_name: folder_with_sr_????.png}
    Creates a grid: rows = images, cols = [HR, Bicubic, model1, model2, ...]
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    model_names = list(result_dirs.keys())
    n_models    = len(model_names)
    n_cols      = 2 + n_models  # HR + Bicubic + models

    fig, axes = plt.subplots(n_images, n_cols,
                             figsize=(3 * n_cols, 3 * n_images))
    if n_images == 1:
        axes = [axes]

    for row in range(n_images):
        idx = f"{row:04d}"
        # HR
        hr_path = os.path.join(list(result_dirs.values())[0], f"hr_{idx}.png")
        if os.path.exists(hr_path):
            axes[row][0].imshow(Image.open(hr_path))
        axes[row][0].set_title("HR" if row == 0 else "")
        axes[row][0].axis("off")

        # Bicubic
        bic_path = os.path.join(list(result_dirs.values())[0], f"bicubic_{idx}.png")
        if os.path.exists(bic_path):
            axes[row][1].imshow(Image.open(bic_path))
        axes[row][1].set_title("Bicubic" if row == 0 else "")
        axes[row][1].axis("off")

        # Models
        for col, (name, folder) in enumerate(result_dirs.items()):
            sr_path = os.path.join(folder, f"sr_{idx}.png")
            if os.path.exists(sr_path):
                axes[row][2 + col].imshow(Image.open(sr_path))
            axes[row][2 + col].set_title(name if row == 0 else "")
            axes[row][2 + col].axis("off")

    fig.suptitle("Visual Super-Resolution Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
