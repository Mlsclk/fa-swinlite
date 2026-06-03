"""
Training loop for FA-SwinLite.
Supports:
  - Single model training
  - Full ablation study (4 configurations)
  - Identical parameter-budget comparison
"""

import os
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from model import FASwinLite
from losses import CombinedLoss
from evaluate import Evaluator, print_results_table, save_results_json


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                sr_batch = model(lr_batch)
                losses   = criterion(sr_batch, hr_batch)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            sr_batch = model(lr_batch)
            losses   = criterion(sr_batch, hr_batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

        total_loss += losses["total"].item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, max_samples=50):
    from evaluate import psnr as calc_psnr, ssim as calc_ssim
    model.eval()
    psnr_vals, ssim_vals = [], []
    for i, (lr, hr) in enumerate(loader):
        if i >= max_samples:
            break
        sr = model(lr.to(device))
        hr = hr.to(device)
        psnr_vals.append(calc_psnr(sr, hr))
        ssim_vals.append(calc_ssim(sr, hr))

    import numpy as np
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


# ──────────────────────────────────────────────────────────────────────────────
# Main trainer
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model_name: str,
                model:      torch.nn.Module,
                train_loader,
                valid_loader,
                use_freq:   bool,
                scale:      int,
                device:     str,
                num_epochs: int = Config.NUM_EPOCHS,
                checkpoint_dir: str = Config.CHECKPOINT_DIR):

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_x{scale}_best.pth")

    model = model.to(device)
    criterion = CombinedLoss(
        lambda_pix  = Config.LAMBDA_PIX,
        lambda_perc = Config.LAMBDA_PERC,
        lambda_freq = Config.LAMBDA_FREQ,
        use_freq    = use_freq,
        device      = device,
    )

    optimizer = optim.Adam(model.parameters(),
                           lr=Config.LR,
                           weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=Config.LR_MIN)
    scaler    = torch.cuda.amp.GradScaler() if device != "cpu" else None

    best_psnr  = 0.0
    log        = []

    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Scale ×{scale} | "
          f"Params: {model.count_parameters()/1e6:.2f}M | "
          f"FreqLoss: {use_freq}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_psnr, val_ssim = validate(model, valid_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:03d}/{num_epochs}] "
              f"Loss: {train_loss:.4f} | "
              f"PSNR: {val_psnr:.2f} dB | "
              f"SSIM: {val_ssim:.4f} | "
              f"LR: {lr_now:.2e} | "
              f"Time: {elapsed:.1f}s")

        log.append({"epoch": epoch, "loss": train_loss,
                    "psnr": val_psnr, "ssim": val_ssim, "lr": lr_now})

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch":       epoch,
                "model_name":  model_name,
                "scale":       scale,
                "state_dict":  model.state_dict(),
                "psnr":        val_psnr,
                "ssim":        val_ssim,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint ({val_psnr:.2f} dB)")

    # Save training log
    log_path = os.path.join(checkpoint_dir, f"{model_name}_x{scale}_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nBest PSNR: {best_psnr:.2f} dB | Checkpoint: {ckpt_path}")
    return best_psnr, ckpt_path


# ──────────────────────────────────────────────────────────────────────────────
# Ablation study runner
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(scale: int,
                 train_loader,
                 valid_loader,
                 device: str,
                 num_epochs: int = Config.NUM_EPOCHS,
                 results_dir: str = Config.RESULTS_DIR):

    os.makedirs(results_dir, exist_ok=True)
    ablation_results = {}

    for name, use_lightweight, use_freq in Config.ABLATION_CONFIGS:
        model = FASwinLite(scale=scale, lightweight=use_lightweight)
        best_psnr, ckpt_path = train_model(
            model_name   = name,
            model        = model,
            train_loader = train_loader,
            valid_loader = valid_loader,
            use_freq     = use_freq,
            scale        = scale,
            device       = device,
            num_epochs   = num_epochs,
        )

        # Full evaluation
        evaluator = Evaluator(device=device,
                              save_dir=os.path.join(results_dir, name))
        # Load best checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = evaluator.evaluate_loader(model, valid_loader, max_samples=100)
        metrics["Params_M"]      = round(model.count_parameters() / 1e6, 2)
        metrics["lightweight"]   = use_lightweight
        metrics["freq_loss"]     = use_freq
        ablation_results[name]   = metrics

    print_results_table(ablation_results)
    save_results_json(ablation_results,
                      os.path.join(results_dir, f"ablation_x{scale}.json"))
    return ablation_results


# ──────────────────────────────────────────────────────────────────────────────
# Lambda_freq sweep (ablation on frequency loss weight)
# ──────────────────────────────────────────────────────────────────────────────

def run_lambda_freq_sweep(scale: int,
                           train_loader,
                           valid_loader,
                           device: str,
                           lambda_values: list = [0.0, 0.05, 0.1, 0.5, 1.0],
                           num_epochs: int = 30,
                           results_dir: str = Config.RESULTS_DIR):
    """Sweep over lambda_freq values for the proposed FA-SwinLite model."""
    sweep_results = {}
    os.makedirs(results_dir, exist_ok=True)

    for lam in lambda_values:
        Config.LAMBDA_FREQ = lam
        name = f"FA_SwinLite_lam{lam}"
        model = FASwinLite(scale=scale, lightweight=True)
        best_psnr, ckpt_path = train_model(
            model_name   = name,
            model        = model,
            train_loader = train_loader,
            valid_loader = valid_loader,
            use_freq     = (lam > 0),
            scale        = scale,
            device       = device,
            num_epochs   = num_epochs,
        )
        evaluator = Evaluator(device=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = evaluator.evaluate_loader(model, valid_loader, max_samples=50)
        metrics["lambda_freq"] = lam
        sweep_results[name]    = metrics
        print(f"  λ_freq={lam:.2f} → PSNR={metrics['PSNR']:.2f} dB")

    save_results_json(sweep_results,
                      os.path.join(results_dir, f"lambda_sweep_x{scale}.json"))
    return sweep_results
