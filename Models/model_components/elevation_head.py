#! /usr/bin/env python3

import torch
import torch.nn as nn


# ── Elevation-bin constants ───────────────────────────────────────────────────
_H_MIN  = -0.50   # metres
_H_MAX  =  1.50   # metres
_H_STEP =  0.05   # metres  (5 cm)
NUM_ELEVATION_BINS = int(round((_H_MAX - _H_MIN) / _H_STEP))   # 40 bins


class ElevationHead(nn.Module):
    """Softmax-based classifier: predicts a probability distribution across elevation bins.

    Bin layout  (40 bins, 5 cm each):
        bin 0  →  centre at −0.475 m  (range −0.500 … −0.450 m)
        bin 1  →  centre at −0.425 m
        …
        bin 39 →  centre at +1.475 m  (range +1.450 … +1.500 m)

    Args:
        in_channels (int): Channel width from ElevationNeck (default 256).
        num_bins    (int): Number of elevation bins (default 40).
    """

    def __init__(self, in_channels: int = 256, num_bins: int = NUM_ELEVATION_BINS):
        super(ElevationHead, self).__init__()
        self.GeLU     = nn.GELU()
        self.num_bins = num_bins

        self.conv_0 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(128, num_bins, kernel_size=1)

        # Non-trainable buffer: centre elevation of each bin (metres)
        bin_centres = torch.arange(num_bins).float() * _H_STEP + _H_MIN + _H_STEP / 2.0
        self.register_buffer('bin_centres', bin_centres)   # (num_bins,)

    def forward(self, neck: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neck (B, in_channels, H, W): ElevationNeck output.

        Returns:
            probs (B, num_bins, H, W): Softmax probability distribution over elevation bins.
        """
        x      = self.GeLU(self.conv_0(neck))   # (B, 128,      H, W)
        logits = self.conv_1(x)                  # (B, num_bins, H, W)
        probs  = logits.softmax(dim=1)           # (B, num_bins, H, W)
        return probs

    @torch.no_grad()
    def expected_elevation(self, probs: torch.Tensor) -> torch.Tensor:
        """Collapse the bin distribution to a scalar elevation map via weighted sum.

        Args:
            probs (B, num_bins, H, W): Output of forward().

        Returns:
            elevation_map (B, H, W): Expected elevation in metres at each spatial location.
        """
        centres       = self.bin_centres.view(1, -1, 1, 1)   # (1, num_bins, 1, 1)
        elevation_map = (probs * centres).sum(dim=1)          # (B, H, W)
        return elevation_map
