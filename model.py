"""
FA-SwinLite — Frequency-Aware Lightweight Transformer for Image Super-Resolution

Architecture:
  LR Input → Shallow Feature Extraction (Conv)
           → N × Lightweight Swin Transformer Blocks (LSTB)
           → Deep Feature Aggregation (Conv)
           → Residual add with shallow features
           → Pixel-Shuffle Upsampling
           → HR Output
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Window partition / reverse
# ──────────────────────────────────────────────────────────────────────────────

def window_partition(x: torch.Tensor, window_size: int):
    """
    x: (B, H, W, C)
    Returns: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """Reverse of window_partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# Window Multi-Head Self-Attention (W-MSA)
# ──────────────────────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size - 1) * (2*window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2,ws,ws
        coords_flat = torch.flatten(coords, 1)   # 2, ws²
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, ws², ws²
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)   # ws², ws²
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv     = nn.Linear(dim, dim * 3, bias=True)
        self.proj    = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size**2, self.window_size**2, -1)
        bias = bias.permute(2, 0, 1).contiguous()   # num_heads, ws², ws²
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Swin Transformer Block (with optional cyclic shift)
# ──────────────────────────────────────────────────────────────────────────────

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 shift_size: int = 0, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
        self.attn_mask = None  # computed lazily per resolution

    def _compute_attn_mask(self, H: int, W: int, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size**2)
        attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask    = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to multiple of window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Compute mask lazily
        attn_mask = self._compute_attn_mask(Hp, Wp, x.device)

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size**2, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        x = x[:, :H, :W, :].contiguous().view(B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Residual Swin Transformer Group (RSTG)
# ──────────────────────────────────────────────────────────────────────────────

class ResidualSwinGroup(nn.Module):
    """2 Swin blocks (W-MSA + SW-MSA) + residual conv."""
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size, shift_size=0,
                                 mlp_ratio=mlp_ratio, dropout=dropout),
            SwinTransformerBlock(dim, num_heads, window_size,
                                 shift_size=window_size//2,
                                 mlp_ratio=mlp_ratio, dropout=dropout),
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, H: int, W: int):
        residual = x
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.transpose(1, 2).view(-1, self.conv.in_channels, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x + residual


# ──────────────────────────────────────────────────────────────────────────────
# FA-SwinLite
# ──────────────────────────────────────────────────────────────────────────────

class FASwinLite(nn.Module):
    """
    Frequency-Aware Lightweight Transformer Super-Resolution Network.

    Args:
        scale:          upscaling factor (2 or 4)
        lightweight:    if True uses Config.NUM_SWIN_BLOCKS and Config.EMBED_DIM
                        if False doubles blocks/dim (full-size baseline)
    """

    def __init__(self, scale: int = 2, lightweight: bool = True, in_channels: int = 3):
        super().__init__()
        self.scale = scale

        dim     = Config.EMBED_DIM   if lightweight else Config.EMBED_DIM * 2
        n_grps  = Config.NUM_SWIN_BLOCKS if lightweight else Config.NUM_SWIN_BLOCKS * 2

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # Deep feature extraction: N residual Swin groups
        self.groups = nn.ModuleList([
            ResidualSwinGroup(dim, Config.NUM_HEADS, Config.WINDOW_SIZE,
                              Config.MLP_RATIO, Config.DROPOUT)
            for _ in range(n_grps)
        ])
        self.norm       = nn.LayerNorm(dim)
        self.conv_after = nn.Conv2d(dim, dim, 3, 1, 1)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
        )
        self.conv_last = nn.Conv2d(dim, in_channels, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Pad so H,W are multiples of window_size
        ws  = Config.WINDOW_SIZE
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.shape

        # Shallow features
        feat = self.conv_first(x)                       # B,dim,Hp,Wp

        # Deep features
        seq = feat.flatten(2).transpose(1, 2)           # B, Hp*Wp, dim
        for grp in self.groups:
            seq = grp(seq, Hp, Wp)
        seq = self.norm(seq)
        deep = seq.transpose(1, 2).view(B, -1, Hp, Wp)
        deep = self.conv_after(deep)

        # Residual + upsample
        out = self.upsample(feat + deep)
        out = self.conv_last(out)

        # Remove padding (scaled)
        out = out[:, :, :H*self.scale, :W*self.scale]
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for scale in [2, 4]:
        for lw in [True, False]:
            m = FASwinLite(scale=scale, lightweight=lw)
            x = torch.randn(1, 3, 64, 64)
            y = m(x)
            label = "Lightweight" if lw else "Full"
            print(f"Scale ×{scale} | {label:10s} | "
                  f"Params: {m.count_parameters()/1e6:.2f}M | "
                  f"Output: {tuple(y.shape)}")
