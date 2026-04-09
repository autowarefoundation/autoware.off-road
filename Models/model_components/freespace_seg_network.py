#! /usr/bin/env python3

import torch.nn as nn
from .backbone          import Backbone
from .scene_context     import SceneContext
from .scene_neck        import SceneNeck
from .freespace_seg_head import FreespaceSegHead


class FreespaceSegNetwork(nn.Module):
    def __init__(self):
        super(FreespaceSegNetwork, self).__init__()

        # Encoder
        self.Backbone = Backbone()

        # Context
        self.SceneContext = SceneContext()

        # Neck
        self.SceneNeck = SceneNeck()

        # Head
        self.FreespaceSegHead = FreespaceSegHead()

    def forward(self, image):
        features     = self.Backbone(image)
        deep_features = features[4]
        context      = self.SceneContext(deep_features)
        neck         = self.SceneNeck(context, features)
        output       = self.FreespaceSegHead(neck, features)
        return output
