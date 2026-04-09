from .pre_trained_backbone import PreTrainedBackbone
from .scene_context import SceneContext
from .scene_neck import SceneNeck
from .freespace_seg_head import FreespaceSegHead
from .freespace_contour_head import FreespaceContourHead
import torch.nn as nn


class FreespaceNetwork(nn.Module):
    def __init__(self, pretrained):
        super(FreespaceNetwork, self).__init__()

        # Encoder
        self.PreTrainedBackbone = PreTrainedBackbone(pretrained)

        # Context
        self.SceneContext = SceneContext()

        # Neck
        self.SceneNeck = SceneNeck()

        # Segmentation Head
        self.FreespaceSegHead = FreespaceSegHead()
        
        # Contour Head
        self.FreespaceContourHead = FreespaceContourHead()

    def forward(self, image):
        features = self.PreTrainedBackbone(image)
        deep_features = features[4]
        context = self.SceneContext(deep_features)
        neck = self.SceneNeck(context, features)
        seg_output = self.FreespaceSegHead(neck, features)
        contour_output = self.FreespaceContourHead(deep_features)
        return seg_output, contour_output
