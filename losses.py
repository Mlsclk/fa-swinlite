"""
Loss functions for FA-SwinLite:
  - L1 pixel loss
  - Perceptual loss (VGG-16 features)
  - Frequency magnitude loss (FFT)
  - Combined loss with configurable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import Config


# ──────────────────────────────────────────────────────────────────────────────
# 1. L1 Pixel Loss
# ──────────────────────────────────────────────────────────────────────────────

class PixelLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Perceptual Loss (VGG-16, relu2_2 features)
# ──────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Use features up to relu2_2 (index 9)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:10])
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat   = self.feature_extractor(self._normalize(pred.clamp(0, 1)))
        target_feat = self.feature_extractor(self._normalize(target.clamp(0, 1)))
        return F.l1_loss(pred_feat, target_feat)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Frequency Magnitude Loss (FFT-based)
# ──────────────────────────────────────────────────────────────────────────────

class FrequencyLoss(nn.Module):
    """
    Computes the L1 difference between the FFT magnitude spectra of
    predicted and target images.

    Optionally returns per-band errors (low / mid / high frequency)
    for the gap analysis required by the paper.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def _fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — convert to grayscale for frequency analysis
        gray = 0.299*x[:, 0] + 0.587*x[:, 1] + 0.114*x[:, 2]  # (B, H, W)
        fft  = torch.fft.fft2(gray, norm="ortho")
        mag  = torch.fft.fftshift(torch.abs(fft))               # (B, H, W)
        return mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mag   = self._fft_magnitude(pred)
        target_mag = self._fft_magnitude(target)
        return F.l1_loss(pred_mag, target_mag) * self.loss_weight

    @torch.no_grad()
    def band_errors(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Returns MAE for low / mid / high frequency bands.
        Used for the frequency-band gap analysis in the paper.
        """
        pred_mag   = self._fft_magnitude(pred)
        target_mag = self._fft_magnitude(target)

        B, H, W = pred_mag.shape
        cy, cx  = H // 2, W // 2

        # Radial distance map
        ys = torch.arange(H, device=pred.device).float() - cy
        xs = torch.arange(W, device=pred.device).float() - cx
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt(yy**2 + xx**2)          # (H, W)

        r_max    = min(cy, cx)
        low_mask = r < r_max * 0.2
        mid_mask = (r >= r_max * 0.2) & (r < r_max * 0.6)
        hi_mask  = r >= r_max * 0.6

        err = torch.abs(pred_mag - target_mag)  # (B, H, W)

        def band_mae(mask):
            return err[:, mask].mean().item()

        return {
            "low":  band_mae(low_mask),
            "mid":  band_mae(mid_mask),
            "high": band_mae(hi_mask),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Combined Loss
# ──────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    L_total = λ_pix * L1  +  λ_perc * L_perceptual  +  λ_freq * L_freq

    Set λ_freq = 0 to disable frequency loss (ablation).
    """

    def __init__(self,
                 lambda_pix:  float = Config.LAMBDA_PIX,
                 lambda_perc: float = Config.LAMBDA_PERC,
                 lambda_freq: float = Config.LAMBDA_FREQ,
                 use_freq:    bool  = True,
                 device:      str   = "cpu"):
        super().__init__()
        self.lambda_pix  = lambda_pix
        self.lambda_perc = lambda_perc
        self.lambda_freq = lambda_freq if use_freq else 0.0

        self.pix_loss  = PixelLoss()
        self.perc_loss = PerceptualLoss().to(device)
        self.freq_loss = FrequencyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l_pix  = self.pix_loss(pred, target)
        l_perc = self.perc_loss(pred, target)
        l_freq = self.freq_loss(pred, target)

        total = (self.lambda_pix  * l_pix
               + self.lambda_perc * l_perc
               + self.lambda_freq * l_freq)

        return {
            "total": total,
            "pix":   l_pix.item(),
            "perc":  l_perc.item(),
            "freq":  l_freq.item(),
        }
