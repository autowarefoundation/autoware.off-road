#! /usr/bin/env python3

import torch.nn as nn


class PreTrainedBackbone(nn.Module):
    def __init__(self, pretrainedModel):
        super(PreTrainedBackbone, self).__init__()

        self.pretrainedBackBone = pretrainedModel.Backbone

    def forward(self, image):
        features = self.pretrainedBackBone(image)
        return features
