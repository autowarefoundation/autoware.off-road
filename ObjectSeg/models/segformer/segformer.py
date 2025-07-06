import torch
import torch.nn as nn
from models.backbone.mit_encoder import MixVisionTransformer, CONF
from models.decoders.segmlp_decoder import MLPDecoder
import lightning as L
from torchmetrics.classification import MulticlassJaccardIndex, Accuracy
import hydra.utils as hu
from omegaconf import OmegaConf
import functools
from datamodule.goose_dataset import GooseDataset

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
    def __init__(self, num_classes: int = 19, variant: str = "B0",optim_config = None,lr_scheduler_config = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegFormer(variant, num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255)
        self.acc  = Accuracy(task="multiclass", num_classes=num_classes,
                                          ignore_index=255)
        self.optim_config = optim_config
        self.lr_scheduler_config = lr_scheduler_config

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
        opt = hu.instantiate(self.optim_config,params=self.parameters())
        sched = hu.instantiate(self.lr_scheduler_config,optimizer=opt)
        if isinstance(sched, functools.partial):
            sched = sched()          # ← turn into real scheduler
        return {"optimizer": opt, "lr_scheduler": sched}

if __name__ == '__main__':
    model = SegFormer(variant='B0', num_classes=64)
    dataset = GooseDataset(root_dir="/home/autokarthik/autoware.off-road/goose-dataset", split="train")
    x,z = dataset[0]
    x = x.unsqueeze(0)
    z = z.unsqueeze(0)
    y = model(x)
    #z = torch.randn(1,64,224,224)
    loss = nn.CrossEntropyLoss()
    print(loss(y,z))
    print(y.shape)