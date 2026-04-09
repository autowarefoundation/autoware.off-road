#! /usr/bin/env python3

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('..')
from data_utils.augmentations import Augmentations
from model_components.elevation_network import ElevationNetwork


# Elevation bin constants — must match elevation_head.py
_H_MIN   = -0.50
_H_STEP  =  0.05
_H_MAX   =  1.50
_NUM_BINS = 40


class ElevationTrainer:
    def __init__(self, checkpoint_path='', learning_rate=0.0001):

        self.image          = None
        self.label          = None    # (H, W) uint8 bin indices, full image resolution
        self.camera_params  = None    # (12,)  float32
        self.image_val      = None
        self.label_val      = None
        self.camera_params_val = None

        self.augmented_image = None
        self.augmented_label = None   # (H, W) uint8 after augmentation, full resolution
        self.augmented_image_val = None
        self.augmented_label_val = None

        self.image_tensor  = None     # (1, 3, H, W)
        self.label_tensor  = None     # (1, H/8, W/8)  long
        self.cam_tensor    = None     # (1, 12)
        self.image_val_tensor = None
        self.label_val_tensor = None
        self.cam_val_tensor   = None

        self.calc_loss = None
        self.prediction = None        # (1, 40, H/8, W/8)  softmax probs

        self.checkpoint_path = checkpoint_path

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for training')

        # Model
        self.model = ElevationNetwork()
        if len(self.checkpoint_path) > 0:
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, weights_only=True))
        self.model = self.model.to(self.device)

        # TensorBoard
        self.writer = SummaryWriter()

        # Optimiser
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Image normalisation
        self.image_loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # ── Data assignment ───────────────────────────────────────────────────────

    def set_data(self, image, label, camera_params):
        self.image         = image
        self.label         = label
        self.camera_params = camera_params

    def set_val_data(self, image_val, label_val, camera_params_val):
        self.image_val         = image_val
        self.label_val         = label_val
        self.camera_params_val = camera_params_val

    # ── Augmentations ─────────────────────────────────────────────────────────

    def apply_augmentations(self, is_train):
        if is_train:
            aug = Augmentations(is_train=True, data_type='DEPTH')
            self.augmented_image, self.augmented_label = aug.applyTransformDepth(
                image=self.image, ground_truth=self.label)
        else:
            aug = Augmentations(is_train=False, data_type='DEPTH')
            self.augmented_image_val, self.augmented_label_val = aug.applyTransformDepth(
                image=self.image_val, ground_truth=self.label_val)

    # ── Tensor loading ────────────────────────────────────────────────────────

    def load_data(self, is_train):
        if is_train:
            # Image → (1, 3, H, W)
            self.image_tensor = self.image_loader(self.augmented_image) \
                                    .unsqueeze(0).to(self.device)

            # Label: full-res (H, W) uint8 → downscale to (H/8, W/8) → (1, H/8, W/8) long
            lbl = self.augmented_label.astype(np.uint8)
            h, w = lbl.shape
            lbl_small = cv2.resize(lbl, (w // 8, h // 8),
                                   interpolation=cv2.INTER_NEAREST)
            self.label_tensor = torch.from_numpy(lbl_small.astype(np.int64)) \
                                     .unsqueeze(0).to(self.device)

            # Camera params → (1, 12)
            self.cam_tensor = torch.from_numpy(self.camera_params) \
                                   .unsqueeze(0).to(self.device)
        else:
            self.image_val_tensor = self.image_loader(self.augmented_image_val) \
                                        .unsqueeze(0).to(self.device)

            lbl_val = self.augmented_label_val.astype(np.uint8)
            h, w = lbl_val.shape
            lbl_val_small = cv2.resize(lbl_val, (w // 8, h // 8),
                                       interpolation=cv2.INTER_NEAREST)
            self.label_val_tensor = torch.from_numpy(lbl_val_small.astype(np.int64)) \
                                         .unsqueeze(0).to(self.device)

            self.cam_val_tensor = torch.from_numpy(self.camera_params_val) \
                                       .unsqueeze(0).to(self.device)

    # ── Forward / loss ────────────────────────────────────────────────────────

    def run_model(self):
        # ElevationHead returns softmax probs → use NLL loss (log-prob input)
        self.prediction = self.model(self.image_tensor, self.cam_tensor)
        log_probs = self.prediction.clamp(min=1e-7).log()     # (1, 40, H/8, W/8)
        self.calc_loss = F.nll_loss(log_probs, self.label_tensor)

    def loss_backward(self):
        self.calc_loss.backward()

    def get_loss(self):
        return self.calc_loss.item()

    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, image_val, label_val, camera_params_val):
        """Returns MAE in metres for one sample."""
        self.set_val_data(image_val, label_val, camera_params_val)
        self.apply_augmentations(is_train=False)
        self.load_data(is_train=False)

        probs = self.model(self.image_val_tensor, self.cam_val_tensor)  # (1, 40, H/8, W/8)
        expected = self.model.ElevationHead.expected_elevation(probs)   # (1, H/8, W/8)

        # Convert GT bin indices to metres (bin centres)
        gt_metres = self.label_val_tensor.float() * _H_STEP + _H_MIN + _H_STEP / 2.0
        mae = (expected - gt_metres).abs().mean().item()
        return mae

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_loss(self, log_count, loss=None):
        if loss is None:
            loss = self.get_loss()
        print('Logging Training Loss', log_count, loss)
        self.writer.add_scalar('Loss/train', loss, log_count)

    def log_MAE(self, mae, log_count):
        print(f'Validation MAE: {mae:.4f} m')
        self.writer.add_scalar('Val/MAE_metres', mae, log_count)

    # ── Visualization ─────────────────────────────────────────────────────────

    def save_visualization(self, log_count):
        print('Saving Visualization')
        with torch.no_grad():
            probs = self.prediction                                      # (1, 40, H/8, W/8)
            elev  = self.model.ElevationHead.expected_elevation(probs)   # (1, H/8, W/8)

        elev_np  = elev.squeeze(0).cpu().numpy()                         # (H/8, W/8)
        pred_vis = self._elevation_to_colormap(elev_np)

        # GT colourmap
        gt_np    = self.label_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        gt_metres = gt_np * _H_STEP + _H_MIN + _H_STEP / 2.0
        gt_vis   = self._elevation_to_colormap(gt_metres)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(self.augmented_image)
        axs[0].set_title('Image',        fontweight='bold')
        axs[1].imshow(gt_vis)
        axs[1].set_title('Ground Truth', fontweight='bold')
        axs[2].imshow(pred_vis)
        axs[2].set_title('Prediction',   fontweight='bold')

        save_path = os.path.join(os.getcwd(), f'vis_elevation_{log_count}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
        self.writer.add_figure('elevation predictions vs. actuals', fig,
                               global_step=log_count)
        plt.close(fig)

    @staticmethod
    def _elevation_to_colormap(elev_map: np.ndarray) -> np.ndarray:
        """Map an elevation array (metres) to an RGB uint8 image via jet colormap."""
        norm = np.clip((elev_map - _H_MIN) / (_H_MAX - _H_MIN), 0.0, 1.0)
        rgb  = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
        return rgb

    # ── Model persistence ─────────────────────────────────────────────────────

    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')
