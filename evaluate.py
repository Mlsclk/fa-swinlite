"""
Evaluation module:
  - PSNR, SSIM, LPIPS
  - Frequency-band error analysis (gap analysis for paper)
  - Per-model results table
  - Visual result saving
"""

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from losses import FrequencyLoss

# Optional: lpips
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[eval] lpips not installed — LPIPS metric will be skipped. "
          "Install with: pip install lpips")


# ──────────────────────────────────────────────────────────────────────────────
# Image quality metrics
# ──────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better. Typical SR values: 28–38 dB.
    """
    mse = F.mse_loss(pred.clamp(0, max_val), target.clamp(0, max_val))
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(max_val**2 / mse)).item()


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Structural Similarity Index (SSIM).
    Range [0,1]; higher is better.
    Computed on luminance channel (Y from YCbCr).
    """
    # Convert to Y channel
    def rgb2y(t):
        r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        return 0.257*r + 0.504*g + 0.098*b + 16/255

    pred_y   = rgb2y(pred.clamp(0, 1))
    target_y = rgb2y(target.clamp(0, 1))

    # Gaussian kernel
    coords  = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)         # ws, ws
    kernel = kernel.unsqueeze(0).unsqueeze(0)         # 1,1,ws,ws

    pad = window_size // 2
    mu1    = F.conv2d(pred_y,   kernel, padding=pad)
    mu2    = F.conv2d(target_y, kernel, padding=pad)
    mu1_sq = mu1 ** 2;  mu2_sq = mu2 ** 2;  mu1_mu2 = mu1 * mu2
    s1     = F.conv2d(pred_y   ** 2, kernel, padding=pad) - mu1_sq
    s2     = F.conv2d(target_y ** 2, kernel, padding=pad) - mu2_sq
    s12    = F.conv2d(pred_y * target_y, kernel, padding=pad) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1) * (2*s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, device: str = "cpu", save_dir: Optional[str] = None):
        self.device    = device
        self.save_dir  = save_dir
        self.freq_loss = FrequencyLoss()

        self.lpips_fn = None
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
            self.lpips_fn.eval()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    @torch.no_grad()
    def evaluate_loader(self, model: torch.nn.Module,
                        loader, max_samples: int = 100) -> dict:
        """
        Run model on a DataLoader, return aggregated metrics dict.
        """
        model.eval()
        scores = {"psnr": [], "ssim": [], "lpips": [],
                  "freq_low": [], "freq_mid": [], "freq_high": []}

        for i, (lr, hr) in enumerate(loader):
            if i >= max_samples:
                break
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            sr = model(lr)

            scores["psnr"].append(psnr(sr, hr))
            scores["ssim"].append(ssim(sr, hr))

            if self.lpips_fn is not None:
                lp = self.lpips_fn(sr.clamp(0, 1)*2 - 1, hr.clamp(0, 1)*2 - 1)
                scores["lpips"].append(lp.item())

            bands = self.freq_loss.band_errors(sr, hr)
            scores["freq_low"].append(bands["low"])
            scores["freq_mid"].append(bands["mid"])
            scores["freq_high"].append(bands["high"])

            # Save first 5 SR images
            if self.save_dir and i < 5:
                save_image(sr.clamp(0, 1), f"{self.save_dir}/sr_{i:04d}.png")
                save_image(hr,             f"{self.save_dir}/hr_{i:04d}.png")
                # Bicubic baseline
                bic = F.interpolate(lr, scale_factor=model.scale, mode="bicubic",
                                    align_corners=False).clamp(0, 1)
                save_image(bic, f"{self.save_dir}/bicubic_{i:04d}.png")

        def avg(lst):
            return float(np.mean(lst)) if lst else None

        return {
            "PSNR":       avg(scores["psnr"]),
            "SSIM":       avg(scores["ssim"]),
            "LPIPS":      avg(scores["lpips"]) if scores["lpips"] else "N/A",
            "Freq_Low":   avg(scores["freq_low"]),
            "Freq_Mid":   avg(scores["freq_mid"]),
            "Freq_High":  avg(scores["freq_high"]),
            "n_samples":  len(scores["psnr"]),
        }

    @torch.no_grad()
    def frequency_band_analysis(self, models_dict: dict, loader,
                                 max_samples: int = 50) -> dict:
        """
        Run frequency-band error analysis across multiple models.
        Returns a dict suitable for plotting (gap analysis figure).

        models_dict: {"ModelName": model, ...}
        """
        results = {}
        for name, model in models_dict.items():
            model.eval()
            bands = {"low": [], "mid": [], "high": []}
            for i, (lr, hr) in enumerate(loader):
                if i >= max_samples:
                    break
                sr = model(lr.to(self.device))
                hr = hr.to(self.device)
                b  = self.freq_loss.band_errors(sr, hr)
                for k in bands:
                    bands[k].append(b[k])
            results[name] = {k: float(np.mean(v)) for k, v in bands.items()}
            print(f"  {name:30s} | low={results[name]['low']:.4f} "
                  f"mid={results[name]['mid']:.4f} high={results[name]['high']:.4f}")
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Results table printer
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(results: dict):
    """Pretty-print a dict of {model_name: metrics_dict}."""
    header = f"{'Model':<30} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} "
    header += f"{'FreqL':>8} {'FreqM':>8} {'FreqH':>8}"
    print("\n" + "="*80)
    print(header)
    print("-"*80)
    for name, m in results.items():
        lpips_str = f"{m['LPIPS']:8.4f}" if isinstance(m['LPIPS'], float) else "     N/A"
        print(f"{name:<30} {m['PSNR']:8.2f} {m['SSIM']:8.4f} {lpips_str} "
              f"{m['Freq_Low']:8.4f} {m['Freq_Mid']:8.4f} {m['Freq_High']:8.4f}")
    print("="*80 + "\n")


def save_results_json(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
