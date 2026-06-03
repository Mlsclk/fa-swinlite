"""
DIV2K Dataset — download, preprocess, and load.
Supports bicubic degradation for ×2 and ×4.
"""

import os
import random
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from config import Config


# ── DIV2K download URLs ────────────────────────────────────────────────────────
DIV2K_URLS = {
    "train_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "train_LR_x2": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
    "valid_LR_x2": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip",
    "train_LR_x4": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "valid_LR_x4": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
}


def _download_and_extract(url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(zip_path):
        print(f"  Downloading {os.path.basename(url)} ...")
        urllib.request.urlretrieve(url, zip_path)
    print(f"  Extracting {os.path.basename(url)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def download_div2k(root: str = Config.DATA_ROOT, scales=(2, 4)):
    """Download DIV2K HR + LR (bicubic) for requested scales."""
    root = Path(root)
    keys = ["train_HR", "valid_HR"]
    for s in scales:
        keys += [f"train_LR_x{s}", f"valid_LR_x{s}"]

    for key in keys:
        url = DIV2K_URLS[key]
        dest = root / key
        marker = dest / ".done"
        if marker.exists():
            print(f"  [skip] {key} already downloaded.")
            continue
        _download_and_extract(url, str(root))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    print("DIV2K download complete.")


def _get_image_paths(folder: str):
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(
        str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts
    )


# ── Dataset ────────────────────────────────────────────────────────────────────

class DIV2KDataset(Dataset):
    """
    Loads paired (LR, HR) patches from DIV2K.
    For training: random crop + augment.
    For validation: center crop.
    """

    def __init__(self, root: str, scale: int, split: str = "train"):
        super().__init__()
        assert split in ("train", "valid")
        assert scale in (2, 4)
        self.scale = scale
        self.split = split
        self.patch_size = Config.PATCH_SIZE   # LR patch

        root = Path(root)
        hr_folder = root / f"{split}_HR" / f"DIV2K_{split}_HR"
        lr_folder = root / f"{split}_LR_x{scale}" / f"DIV2K_{split}_LR_bicubic" / f"X{scale}"

        self.hr_paths = _get_image_paths(str(hr_folder))
        self.lr_paths = _get_image_paths(str(lr_folder))

        assert len(self.hr_paths) == len(self.lr_paths), (
            f"HR/LR count mismatch: {len(self.hr_paths)} vs {len(self.lr_paths)}"
        )
        assert len(self.hr_paths) > 0, f"No images found in {hr_folder}"

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_paths[idx]).convert("RGB")
        hr = Image.open(self.hr_paths[idx]).convert("RGB")

        if self.split == "train":
            lr, hr = self._random_crop(lr, hr)
            lr, hr = self._augment(lr, hr)
        else:
            lr, hr = self._center_crop(lr, hr)

        lr = TF.to_tensor(lr)   # [3, H, W]  float32 [0,1]
        hr = TF.to_tensor(hr)
        return lr, hr

    # ── Crop helpers ────────────────────────────────────────────────────────

    def _random_crop(self, lr: Image.Image, hr: Image.Image):
        lw, lh = lr.size
        ps = self.patch_size
        if lw < ps or lh < ps:
            lr = TF.resize(lr, (max(ps, lh), max(ps, lw)))
            lw, lh = lr.size
        x0 = random.randint(0, lw - ps)
        y0 = random.randint(0, lh - ps)
        lr = lr.crop((x0, y0, x0 + ps, y0 + ps))
        s  = self.scale
        hr = hr.crop((x0*s, y0*s, (x0+ps)*s, (y0+ps)*s))
        return lr, hr

    def _center_crop(self, lr: Image.Image, hr: Image.Image):
        lw, lh = lr.size
        ps = min(lw, lh, 128)          # larger patch for validation
        x0 = (lw - ps) // 2
        y0 = (lh - ps) // 2
        lr = lr.crop((x0, y0, x0+ps, y0+ps))
        s  = self.scale
        hr = hr.crop((x0*s, y0*s, (x0+ps)*s, (y0+ps)*s))
        return lr, hr

    # ── Augmentation (flip + rotate) ────────────────────────────────────────

    def _augment(self, lr, hr):
        if random.random() < 0.5:
            lr = TF.hflip(lr); hr = TF.hflip(hr)
        if random.random() < 0.5:
            lr = TF.vflip(lr); hr = TF.vflip(hr)
        k = random.randint(0, 3)
        if k > 0:
            lr = TF.rotate(lr, 90*k); hr = TF.rotate(hr, 90*k)
        return lr, hr


# ── Convenience builders ───────────────────────────────────────────────────────

def get_loaders(scale: int, root: str = Config.DATA_ROOT):
    train_ds = DIV2KDataset(root, scale, split="train")
    valid_ds = DIV2KDataset(root, scale, split="valid")
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print(f"Scale ×{scale} — train: {len(train_ds)}, valid: {len(valid_ds)}")
    return train_loader, valid_loader
