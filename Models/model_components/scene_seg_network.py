from .backbone import Backbone
from .scene_context import SceneContext
from .scene_neck import SceneNeck
from .scene_seg_head import SceneSegHead
import torch.nn as nn


class SceneSegNetwork(nn.Module):
    def __init__(self):
        super(SceneSegNetwork, self).__init__()

        # Encoder
        self.Backbone = Backbone()

        # Context
        self.SceneContext = SceneContext()

        # Neck
        self.SceneNeck = SceneNeck()

        # Head
        self.SceneSegHead = SceneSegHead()

    def forward(self, image):
        features = self.Backbone(image)
        deep_features = features[4]
        context = self.SceneContext(deep_features)
        neck = self.SceneNeck(context, features)
        output = self.SceneSegHead(neck, features)
        return output
