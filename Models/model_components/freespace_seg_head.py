#! /usr/bin/env python3

import torch.nn as nn


class FreespaceSegHead(nn.Module):
    def __init__(self, num_classes=2):
        super(FreespaceSegHead, self).__init__()

        self.GeLU = nn.GELU()

        # Decoder
        self.upsample_layer_3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_3 = nn.Conv2d(32, 256, 1)
        self.decode_layer_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_7 = nn.Conv2d(256, 128, 3, 1, 1)

        self.upsample_layer_4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_9 = nn.Conv2d(128, 64, 3, 1, 1)

        # FINAL LOGITS LAYER — 2 classes (non-drivable, drivable)
        self.decode_layer_10 = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, neck, features):

        # Upsample + skip
        d7 = self.upsample_layer_3(neck)
        d7 = d7 + self.skip_link_layer_3(features[0])
        d7 = self.GeLU(self.decode_layer_6(d7))
        d8 = self.GeLU(self.decode_layer_7(d7))

        # Final upsample
        d8 = self.upsample_layer_4(d8)
        d8 = self.GeLU(self.decode_layer_8(d8))
        d9 = self.decode_layer_9(d8)
        d10 = self.GeLU(d9)

        # Logits: (N, 2, H, W)
        output = self.decode_layer_10(d10)

        return output
