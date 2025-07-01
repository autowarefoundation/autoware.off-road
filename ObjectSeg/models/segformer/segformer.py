import torch
import torch.nn as nn
from models.backbone.mit_encoder import MixVisionTransformer, CONF
from models.decoders.segmlp_decoder import MLPDecoder
import lightning as L
from torchmetrics.classification import MulticlassJaccardIndex, Accuracy
import hydra.utils as hu
from omegaconf import OmegaConf


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

class LitSegFormer(L.LightningModule):
    def __init__(self, num_classes: int = 19, variant: str = "B0", lr: float = 6e-5, weight_decay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegFormer(variant, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255)
        self.acc  = Accuracy(task="multiclass", num_classes=num_classes,
                                          ignore_index=255)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        preds = logits.argmax(1)
        self.log(f"{stage}_loss", loss,  prog_bar=True, sync_dist=True)
        self.log(f"{stage}_miou", self.miou(preds, masks), prog_bar=True,
                 sync_dist=True)
        self.log(f"{stage}_acc",  self.acc(preds, masks),  prog_bar=False,
                 sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=160_000,
                                                      power=1.0)  # poly decay
        return {"optimizer": opt, "lr_scheduler": sched}

if __name__ == '__main__':
    model = SegFormer(variant='B0', num_classes=12)
    x = torch.randn(1,3,224,224)
    y = model(x)
    print(y.shape)