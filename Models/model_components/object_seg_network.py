#! /usr/bin/env python3

import torch.nn as nn
from .pre_trained_backbone import PreTrainedBackbone
from .scene_context        import SceneContext
from .scene_neck           import SceneNeck
from .object_seg_head      import ObjectSegHead


class ObjectSegNetwork(nn.Module):
    def __init__(self, pretrained):
        super(ObjectSegNetwork, self).__init__()

        # Encoder
        self.PreTrainedBackbone = PreTrainedBackbone(pretrained)

        # Context
        self.SceneContext = SceneContext()

        # Neck
        self.SceneNeck = SceneNeck()

        # Head — 5 classes
        self.ObjectSegHead = ObjectSegHead(num_classes=5)

    def forward(self, image):
        features     = self.PreTrainedBackbone(image)
        deep_features = features[4]
        context      = self.SceneContext(deep_features)
        neck         = self.SceneNeck(context, features)
        output       = self.ObjectSegHead(neck, features)
        return output
