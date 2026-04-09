#! /usr/bin/env python3

import torch.nn as nn
from .backbone               import Backbone
from .freespace_contour_head import FreespaceContourHead


class FreespaceContourNetwork(nn.Module):
    def __init__(self):
        super(FreespaceContourNetwork, self).__init__()

        # Encoder
        self.Backbone = Backbone()

        # Head — operates directly on deep backbone features (1280ch, stride-32)
        self.FreespaceContourHead = FreespaceContourHead()

    def forward(self, image):
        features     = self.Backbone(image)
        deep_features = features[4]                       # (B, 1280, H/32, W/32)
        output       = self.FreespaceContourHead(deep_features)
        return output
