#! /usr/bin/env python3

import torch.nn as nn
from .backbone           import Backbone
from .scene_context      import SceneContext
from .scene_neck         import SceneNeck
from .elevation_neck     import ElevationNeck
from .elevation_head     import ElevationHead


class ElevationNetwork(nn.Module):
    """Elevation Map perception pipeline — Option 3: SceneNeck shared.

    Shares Backbone + SceneContext + SceneNeck with the freespace pipeline so
    those three stages run once and their output is branched to both tasks.
    ElevationContext (ASPP) is replaced by a lightweight camera-param MLP that
    fuses extrinsic geometry directly into the stride-4 neck features.

    Stage               Component               Role
    ─────────────────────────────────────────────────────────────────────────
    Shared backbone     EfficientNet-B0         Extracts multi-scale features
                                                at strides 2/4/8/16/32.
    Shared context      SceneContext             Global attention on stride-32
                                                deep features.
    Shared neck         SceneNeck               FPN decoder to stride-4 (256ch).
    Camera MLP          Extrinsic MLP           Fuses camera params into neck
                                                features via additive broadcast.
    Elevation Neck      Vertical 1D-Conv        Aggregates vertical pixels into a
                                                geometry-ready latent space.
    Elevation Head      Softmax Classifier      Predicts probability distribution
                                                across 40 elevation bins
                                                (−0.5 … +1.5 m at 5 cm intervals).
    ─────────────────────────────────────────────────────────────────────────

    Args:
        extrinsic_dim (int): Length of the camera-parameter vector.
                             Default 12 = flattened 3×4 [R|t].

    Inputs:
        image         (B, 3, H, W)        RGB image.
        camera_params (B, extrinsic_dim)  Flattened intrinsic / extrinsic
                                          camera parameters.

    Output:
        elevation_probs (B, 40, H/4, W/4) Softmax distribution over elevation
                                          bins.  For a 320×640 input this is
                                          (B, 40, 80, 160).  Call
                                          ``ElevationHead.expected_elevation(probs)``
                                          to collapse to a scalar elevation map
                                          in metres.
    """

    # SceneNeck outputs 256 channels at stride-4
    _NECK_CH = 256

    def __init__(self, extrinsic_dim: int = 12):
        super(ElevationNetwork, self).__init__()

        # ── Shared stages (same architecture as FreespaceSegNetwork) ──────────
        self.Backbone     = Backbone()
        self.SceneContext = SceneContext()
        self.SceneNeck    = SceneNeck()

        # ── Camera-param MLP: projects extrinsics → neck channel space ────────
        self.camera_mlp = nn.Sequential(
            nn.Linear(extrinsic_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, self._NECK_CH),
        )

        self.GeLU = nn.GELU()

        # ── Elevation Neck: Vertical 1D-Conv at stride-8 ──────────────────────
        self.ElevationNeck = ElevationNeck(in_channels=self._NECK_CH,
                                           out_channels=self._NECK_CH)

        # ── Elevation Head: Softmax Classifier ────────────────────────────────
        self.ElevationHead = ElevationHead(in_channels=self._NECK_CH)

    def forward(self, image, camera_params):
        # ── Shared: Backbone ──────────────────────────────────────────────────
        features      = self.Backbone(image)          # [l0, l2, l3, l4, l8]
        deep_features = features[4]                   # (B, 1280, H/32, W/32)

        # ── Shared: Scene Context ─────────────────────────────────────────────
        context = self.SceneContext(deep_features)    # (B, 1280, H/32, W/32)

        # ── Shared: Scene Neck ────────────────────────────────────────────────
        neck = self.SceneNeck(context, features)      # (B, 256,  H/8,  W/8)

        # ── Camera-param fusion: broadcast MLP embedding over spatial dims ────
        cam = self.camera_mlp(camera_params)          # (B, 256)
        cam = cam.unsqueeze(-1).unsqueeze(-1)          # (B, 256, 1, 1)
        cam = cam.expand_as(neck)                     # (B, 256, H/8, W/8)
        neck = self.GeLU(neck + cam)                  # (B, 256, H/8, W/8)

        # ── Elevation Neck ────────────────────────────────────────────────────
        elev_neck = self.ElevationNeck(neck)           # (B, 256, H/8, W/8)

        # ── Elevation Head ────────────────────────────────────────────────────
        elevation_probs = self.ElevationHead(elev_neck)  # (B, 40, H/8, W/8)

        return elevation_probs
