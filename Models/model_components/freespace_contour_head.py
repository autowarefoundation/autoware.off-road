#! /usr/bin/env python3

import torch
import torch.nn as nn


class FreespaceContourHead(nn.Module):
    def __init__(self, num_rays=37, num_bins=46):
        super(FreespaceContourHead, self).__init__()

        self.num_rays = num_rays
        self.num_bins = num_bins
        input_channels = 1280  # Inferred from SceneContext

        self.features_conv = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*10*20, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_rays * num_bins)
        )

    def forward(self, x):
        x = self.features_conv(x)
        x = self.regressor(x)
        x = x.view(-1, self.num_rays, self.num_bins, 1)
        return x
