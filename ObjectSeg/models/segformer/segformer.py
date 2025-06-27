import torch
import torch.nn as nn
from models.backbone.mit_encoder import MixVisionTransformer, CONF
from models.decoders.segmlp_decoder import MLPDecoder


class SegFormer(nn.Module):
    """
    SegFormer
    Args:
        variant: variant of the model (default: 'B0')
        num_classes: number of classes (default: 12)
        decoder_dim: dimension of the decoder (default: 256)
    """
    def __init__(self, variant='B0', num_classes=12, decoder_dim=256):
        super().__init__()
        self.backbone = MixVisionTransformer(variant)
        dims = CONF[variant]["C"]
        self.decode_head = MLPDecoder(dims, decoder_dim, num_classes)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: input image (B, 3, H, W)
        Returns:
            seg: segmentation map (B, num_classes, H, W)
        """
        outs = self.backbone(x)          # 4-scale features
        seg  = self.decode_head(outs)    # B, num_classes, H/4, W/4
        seg  = nn.functional.interpolate(seg, size=x.shape[-2:], mode='bilinear',
                                         align_corners=False)
        return seg

if __name__ == '__main__':
    model = SegFormer(variant='B0', num_classes=12)
    x = torch.randn(1,3,224,224)
    y = model(x)
    print(y.shape)