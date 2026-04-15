#! /usr/bin/env python3

import os
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
from data_utils.augmentations import Augmentations
from data_utils.load_data_object_seg import COLOR_MAP, NUM_CLASSES
from model_components.object_seg_network import ObjectSegNetwork
from model_components.scene_seg_network import SceneSegNetwork


# BGR LUT for fast numpy-based colorization (indexed by class id)
_BGR_LUT = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
for _cid, _rgb in COLOR_MAP.items():
    _BGR_LUT[_cid] = [_rgb[2], _rgb[1], _rgb[0]]  # RGB -> BGR

CLASS_NAMES = ['background', 'man-made_structures', 'vulnerable_living', 'vehicle', 'natural_obstacles']


class ObjectSegTrainer:
    def __init__(self, checkpoint_path='',
                 pretrained_checkpoint_path='',
                 is_pretrained=False,
                 learning_rate=0.0001):

        self.image = 0
        self.image_val = 0
        self.gt = 0
        self.gt_val = 0
        self.class_weights = 0
        self.gt_fused = 0
        self.gt_val_fused = 0
        self.augmented = 0
        self.augmented_val = 0
        self.image_tensor = 0
        self.image_val_tensor = 0
        self.gt_tensor = 0
        self.gt_val_tensor = 0
        self.class_weights_tensor = 0
        self.loss = 0
        self.prediction = 0
        self.calc_loss = 0
        self.checkpoint_path = checkpoint_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for training')

        scene_backbone = SceneSegNetwork()
        self.model = ObjectSegNetwork(scene_backbone)

        if len(pretrained_checkpoint_path) > 0 and not is_pretrained:
            # Load scene backbone weights only
            pretrained_dict = torch.load(pretrained_checkpoint_path, weights_only=True,
                                         map_location=self.device)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f'Loaded pretrained backbone from {pretrained_checkpoint_path}')

        if is_pretrained and len(checkpoint_path) > 0:
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True,
                                                   map_location=self.device))
            print(f'Loaded checkpoint from {checkpoint_path}')

        self.model = self.model.to(self.device)

        self.writer = SummaryWriter()

        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        self.image_loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # ------------------------------------------------------------------
    # Helpers: encode/decode color mask <-> single-channel label indices
    # ------------------------------------------------------------------

    @staticmethod
    def encode_color_mask(color_mask):
        """(H,W,3) RGB -> (H,W) uint8 class indices."""
        h, w, _ = color_mask.shape
        out = np.zeros((h, w), dtype=np.uint8)
        for class_id, rgb in COLOR_MAP.items():
            out[np.all(color_mask == rgb, axis=-1)] = class_id
        return out

    @staticmethod
    def decode_class_mask(index_mask):
        """(H,W) class indices -> (H,W,3) RGB color mask."""
        h, w = index_mask.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, rgb in COLOR_MAP.items():
            out[index_mask == class_id] = rgb
        return out

    # ------------------------------------------------------------------
    # Data assignment
    # ------------------------------------------------------------------

    def set_data(self, image, gt, class_weights):
        self.image = image
        self.gt = gt
        self.class_weights = class_weights

    def set_val_data(self, image_val, gt_val):
        self.image_val = image_val
        self.gt_val = gt_val

    # ------------------------------------------------------------------
    # Augmentations
    # ------------------------------------------------------------------

    def apply_augmentations(self, is_train):
        if is_train:
            aug = Augmentations(is_train=True, data_type='SEGMENTATION')
            self.gt[0] = self.encode_color_mask(self.gt[0])
            self.image, self.augmented = aug.applyTransformSeg(
                image=self.image, ground_truth=self.gt)
            self.gt[0] = self.decode_class_mask(self.gt[0])
            self.augmented[0] = self.decode_class_mask(self.augmented[0])
            # gt_fused: (H,W,NUM_CLASSES) stacking per-class binary masks
            self.gt_fused = np.stack(
                [self.augmented[i + 1] for i in range(NUM_CLASSES)], axis=2)
        else:
            aug = Augmentations(is_train=False, data_type='SEGMENTATION')
            self.gt_val[0] = self.encode_color_mask(self.gt_val[0])
            self.image_val, self.augmented_val = aug.applyTransformSeg(
                image=self.image_val, ground_truth=self.gt_val)
            self.gt_val[0] = self.decode_class_mask(self.gt_val[0])
            self.augmented_val[0] = self.decode_class_mask(self.augmented_val[0])
            self.gt_val_fused = np.stack(
                [self.augmented_val[i + 1] for i in range(NUM_CLASSES)], axis=2)

    # ------------------------------------------------------------------
    # Tensor loading
    # ------------------------------------------------------------------

    def load_data(self, is_train):
        self._load_image_tensor(is_train)
        self._load_gt_tensor(is_train)
        if is_train:
            self.class_weights_tensor = torch.tensor(
                self.class_weights, dtype=torch.float32).to(self.device)

    def _load_image_tensor(self, is_train):
        if is_train:
            t = self.image_loader(self.image).unsqueeze(0).to(self.device)
            self.image_tensor = t
        else:
            t = self.image_loader(self.image_val).unsqueeze(0).to(self.device)
            self.image_val_tensor = t

    def _load_gt_tensor(self, is_train):
        if is_train:
            gt_idx = np.argmax(self.gt_fused, axis=2).astype(np.int64)
            self.gt_tensor = torch.from_numpy(gt_idx).unsqueeze(0).to(self.device)
        else:
            gt_idx = np.argmax(self.gt_val_fused, axis=2).astype(np.int64)
            self.gt_val_tensor = torch.from_numpy(gt_idx).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def run_model(self):
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        self.prediction = self.model(self.image_tensor)
        self.calc_loss = self.loss(self.prediction, self.gt_tensor)

    def loss_backward(self):
        self.calc_loss.backward()

    def get_loss(self):
        return self.calc_loss.item()

    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Validation / IoU
    # ------------------------------------------------------------------

    def IoU(self, output, label):
        intersection = np.logical_and(label, output)
        union = np.logical_or(label, output)
        return (np.sum(intersection) + 1) / float(np.sum(union) + 1)

    def calc_IoU_val(self):
        output_val = self.model(self.image_val_tensor)        # (1,C,H,W)
        pred = torch.argmax(output_val, dim=1)                # (1,H,W)
        one_hot = torch.nn.functional.one_hot(
            pred, num_classes=NUM_CLASSES).squeeze(0)         # (H,W,C)
        one_hot = one_hot.cpu().detach().numpy()

        iou_full = self.IoU(one_hot, self.gt_val_fused)
        iou_per_class = [
            self.IoU(one_hot[..., i], self.gt_val_fused[..., i])
            for i in range(NUM_CLASSES)
        ]
        return iou_full, iou_per_class

    def validate(self, image_val, gt_val):
        self.set_val_data(image_val, gt_val)
        self.apply_augmentations(is_train=False)
        self.load_data(is_train=False)
        return self.calc_IoU_val()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_loss(self, log_count, loss=None):
        if loss is None:
            loss = self.get_loss()
        print('Logging Training Loss', log_count, loss)
        self.writer.add_scalar('Loss/train', loss, log_count)

    def log_IoU(self, iou_full, iou_per_class, log_count):
        print('Logging Validation IoU')
        self.writer.add_scalar('Val/mIoU', iou_full, log_count)
        self.writer.add_scalars('Val/IoU_Classes',
            {CLASS_NAMES[i]: iou_per_class[i] for i in range(NUM_CLASSES)},
            log_count)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def make_visualization(self, prediction_tensor):
        """Return RGB numpy array (H,W,3) from logit tensor (1,C,H,W)."""
        pred = prediction_tensor.squeeze(0).cpu().detach()   # (C,H,W)
        pred = pred.permute(1, 2, 0)                         # (H,W,C)
        class_idx = torch.argmax(pred, dim=2).numpy()        # (H,W)
        return _BGR_LUT[class_idx][..., ::-1].copy()         # BGR->RGB

    def save_visualization(self, log_count):
        print('Saving Visualization')
        vis_predict = self.make_visualization(self.prediction)
        label = self.augmented[0]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(self.image)
        axs[0].set_title('Image', fontweight='bold')
        axs[1].imshow(label)
        axs[1].set_title('Ground Truth', fontweight='bold')
        axs[2].imshow(vis_predict)
        axs[2].set_title('Prediction', fontweight='bold')

        save_path = os.path.join(os.getcwd(), f'vis_{log_count}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
        self.writer.add_figure('predictions vs. actuals', fig, global_step=log_count)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Mode / checkpoint
    # ------------------------------------------------------------------

    def set_train_mode(self):
        self.model = self.model.train()

    def set_eval_mode(self):
        self.model = self.model.eval()

    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')
