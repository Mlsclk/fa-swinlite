# FA-SwinLite: Frequency-Aware Lightweight Transformer for Image Super-Resolution
**ECE 531 Computer Vision — Term Project**
AGU Department of Electrical and Computer Engineering, 2025–2026 Spring
**Author:** Melisa ÇOLAK

---

## Overview
FA-SwinLite is a lightweight Swin Transformer-based super-resolution model enhanced with a **frequency-domain loss** to better restore high-frequency textures. It addresses a well-documented weakness in existing SR models: high-frequency reconstruction errors.

### Key Contributions
- Lightweight SwinIR-style architecture (~fewer params than full SwinIR)
- Frequency-aware loss: `L = λ_pix·L1 + λ_perc·L_perceptual + λ_freq·L_freq`
- Frequency-band gap analysis demonstrating where existing models fail
- Full ablation separating architecture vs. loss contributions
- Identical parameter-budget comparison

---

## Project Structure

fa-swinlite/
├── src/
│   ├── config.py        # All hyperparameters
│   ├── dataset.py       # DIV2K download + dataloader
│   ├── model.py         # FA-SwinLite architecture
│   ├── losses.py        # L1 + Perceptual + Frequency loss
│   ├── train.py         # Training loop + ablation runner
│   ├── evaluate.py      # PSNR, SSIM, LPIPS, band analysis
│   └── visualize.py     # Paper figure generation
├── FA_SwinLite_ECE531.ipynb   # Main Colab notebook
└── README.md

---

## Results (DIV2K ×2)

| Model | PSNR | SSIM | Params |
|-------|------|------|--------|
| Baseline_Full_NoFreq | 28.94 dB | 0.8925 | 1.83M |
| Baseline_Full_Freq | 29.23 dB | 0.8960 | 1.83M |
| FA_SwinLite_NoFreq | 25.78 dB | 0.8529 | 0.29M |
| **FA_SwinLite_Full** | **26.92 dB** | **0.8749** | **0.29M** |

---

## Ablation Study Design

| Config | Lightweight Arch | Frequency Loss |
|--------|:---:|:---:|
| Baseline_Full_NoFreq | No | No |
| Baseline_Full_Freq | No | Yes |
| FA_SwinLite_NoFreq | Yes | No |
| FA_SwinLite_Full | Yes | Yes |

---

## Quick Start (Google Colab)

1. Upload the `src/` folder and `FA_SwinLite_ECE531.ipynb` to Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order

Or clone from GitHub:

git clone https://github.com/Mlsclk/fa-swinlite.git
cd fa-swinlite

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embed dim (lightweight) | 60 |
| Swin blocks | 4 |
| Window size | 8 |
| Batch size | 8 |
| Learning rate | 1e-4 (cosine decay) |
| λ_pix | 1.0 |
| λ_perc | 0.1 |
| λ_freq | 0.1 (swept: 0, 0.05, 0.1, 0.5, 1.0) |
| Scales | ×2, ×4 |
| Dataset | DIV2K (800 train / 100 val) |

---

## Loss Formulation

L_total = λ_pix · L1(SR, HR) + λ_perc · L_perc(VGG(SR), VGG(HR)) + λ_freq · L_freq(|FFT(SR)|, |FFT(HR)|)

---

## Evaluation Metrics

- **PSNR** (dB) — pixel-level fidelity
- **SSIM** — structural similarity
- **LPIPS** — perceptual quality
- **Frequency-band MAE** (low/mid/high) — gap analysis

---

## Colab Notebook

https://colab.research.google.com/drive/1LKll7TxdNedntyk89ZewOZ8qe5AhU6cG

---

## AI Acknowledgment

Claude (Anthropic) was used to assist with code structure. All content has been reviewed and verified by the author, who takes full responsibility for its accuracy and scientific validity.

---

## References

- Lim et al. (2017). EDSR. CVPR Workshops.
- Wang et al. (2018). ESRGAN. ECCV Workshops.
- Liang et al. (2021). SwinIR. ICCV Workshops.
- Agustsson & Timofte (2017). DIV2K. CVPR Workshops.
