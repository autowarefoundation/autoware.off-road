#! /usr/bin/env python3

import torch
import torch.nn as nn


class ElevationNeck(nn.Module):
    """Vertical 1D-Conv: aggregates vertical pixels into a geometry-ready latent space.

    Each image column is treated as an independent 1-D sequence along the height axis.
    Three stacked Conv1d layers (with increasing receptive field) compress the vertical
    cues without touching the horizontal layout, preserving the column-wise structure
    needed for subsequent BEV lifting.

    Args:
        in_channels  (int): Input channel width from ElevationContext (default 512).
        out_channels (int): Output channel width fed to ElevationHead (default 256).
    """

    def __init__(self, in_channels: int = 512, out_channels: int = 256):
        super(ElevationNeck, self).__init__()
        self.GeLU = nn.GELU()

        # Three vertical 1-D conv layers; kernel sizes 3 / 3 / 5 give receptive
        # field 1 + 2 + 2 + 4 = 9 rows while keeping spatial H unchanged.
        self.vconv_0 = nn.Conv1d(in_channels,  in_channels,  kernel_size=3, padding=1)
        self.vconv_1 = nn.Conv1d(in_channels,  out_channels, kernel_size=3, padding=1)
        self.vconv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (B, in_channels, H, W): ElevationContext output.

        Returns:
            out (B, out_channels, H, W): Geometry-ready latent features.
        """
        B, C, H, W = x.shape

        # ── Reshape: each column becomes an independent 1-D sequence ─────────
        # (B, C, H, W) → permute → (B, W, C, H) → reshape → (B*W, C, H)
        x = x.permute(0, 3, 1, 2).reshape(B * W, C, H)

        # ── Vertical 1-D convolutions ─────────────────────────────────────────
        x = self.GeLU(self.vconv_0(x))    # (B*W, in_ch,  H)
        x = self.GeLU(self.vconv_1(x))    # (B*W, out_ch, H)
        x = self.GeLU(self.vconv_2(x))    # (B*W, out_ch, H)

        # ── Restore spatial layout ────────────────────────────────────────────
        out_ch = x.shape[1]
        # (B*W, out_ch, H) → reshape → (B, W, out_ch, H) → permute → (B, out_ch, H, W)
        x = x.reshape(B, W, out_ch, H).permute(0, 2, 3, 1)
        return x
