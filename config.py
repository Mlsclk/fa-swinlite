"""
FA-SwinLite Configuration
ECE 531 Computer Vision - Term Project
"""

class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    DATA_ROOT       = "./data/DIV2K"
    SCALE_FACTORS   = [2, 4]          # upscaling factors to train/eval
    PATCH_SIZE      = 64              # LR patch size during training
    NUM_WORKERS     = 2

    # ── Model ─────────────────────────────────────────────────────────────────
    EMBED_DIM       = 60              # lightweight: SwinIR-small uses 60
    NUM_HEADS       = 6
    WINDOW_SIZE     = 8
    NUM_SWIN_BLOCKS = 4               # reduced from SwinIR's 6 for lightweight
    MLP_RATIO       = 2.0
    DROPOUT         = 0.0

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE      = 16
    NUM_EPOCHS      = 100             # reduce to 30 for quick Colab test
    LR              = 1e-4
    LR_MIN          = 1e-6
    WEIGHT_DECAY    = 1e-4
    GRAD_CLIP       = 1.0

    # ── Loss weights (ablation sweeps over LAMBDA_FREQ) ───────────────────────
    LAMBDA_PIX      = 1.0             # L1 pixel loss weight
    LAMBDA_PERC     = 0.1             # perceptual loss weight
    LAMBDA_FREQ     = 0.1             # frequency magnitude loss weight
                                      # ablation: [0.0, 0.05, 0.1, 0.5, 1.0]

    # ── Ablation experiment names ─────────────────────────────────────────────
    # Each tuple: (name, use_lightweight, use_freq_loss)
    ABLATION_CONFIGS = [
        ("Baseline_Full_NoFreq",   False, False),
        ("Baseline_Full_Freq",     False, True),
        ("FA_SwinLite_NoFreq",     True,  False),
        ("FA_SwinLite_Full",       True,  True),   # proposed method
    ]

    # ── Evaluation ────────────────────────────────────────────────────────────
    EVAL_SCALE      = 2               # primary eval scale
    SAVE_IMAGES     = True

    # ── Paths ─────────────────────────────────────────────────────────────────
    CHECKPOINT_DIR  = "./checkpoints"
    RESULTS_DIR     = "./results"
    LOG_DIR         = "./logs"
