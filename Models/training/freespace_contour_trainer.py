import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
from data_utils.augmentations import Augmentations
from model_components.freespace_networks import FreespaceNetwork
from model_components.scene_seg_network import SceneSegNetwork
from data_utils.radial_contour_dists import RadialContourDists
from data_utils.radial_contour_dists import RadialContourDists
import os
import cv2


class FreespaceContourTrainer():
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
        self.gt_tensor = 0          # (N,H,W) long for CrossEntropyLoss
        self.gt_val_tensor = 0      # (N,H,W) long for CrossEntropyLoss
        self.gt_contour_tensor = 0
        self.gt_val_contour_tensor = 0
        self.class_weights_tensor = 0
        self.loss = 0
        self.contour_loss = 0
        self.prediction = 0
        self.contour_prediction = 0
        self.calc_loss = 0
        self.prediction_vis = 0
        self.checkpoint_path = checkpoint_path

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        # Instantiate model
        # If SceneSegNetwork outputs C=3+, switch to the correct 2-class model/head.
        scene_backbone = SceneSegNetwork()
        self.model = FreespaceNetwork(scene_backbone)
        self.model = self.model.to(self.device)

        if len(self.checkpoint_path) > 0:
            self.model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))
        self.model = self.model.to(self.device)

        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )

        # (Kept, but we won't use ToTensor() for GT anymore, since CE wants (H,W) indices)
        self.gt_loader = transforms.Compose([transforms.ToTensor()])

    # Logging Training Loss
    def log_loss(self, log_count, loss=None, contour_loss=None):
        if loss is None:
            loss = self.get_loss()
        print('Logging Training Loss', log_count, loss, contour_loss)
        self.writer.add_scalar("Loss/train", loss, (log_count))
        if contour_loss is not None:
            self.writer.add_scalar("Loss/train_contour", contour_loss, (log_count))

    # Logging Validation Contour Metrics
    def log_contour_metrics(self, accuracy, mae, log_count):
        print('Logging Validation')
        self.writer.add_scalar("Val/Contour_Accuracy", accuracy, (log_count))
        self.writer.add_scalar("Val/Contour_MAE", mae, (log_count))

    # Assign input variables
    def set_data(self, image, gt, class_weights):
        self.image = image
        self.gt = gt
        self.class_weights = class_weights

    def set_val_data(self, image_val, gt_val):
        self.image_val = image_val
        self.gt_val = gt_val

    def encode_color_mask_to_single_channel(self, color_mask, color_map):
        """
        Convert a color-coded segmentation mask (H, W, 3) to a single-channel label mask (H, W).
        """
        h, w, _ = color_mask.shape
        single_channel = np.zeros((h, w), dtype=np.uint8)

        for class_idx, rgb in color_map.items():
            matches = np.all(color_mask == rgb, axis=-1)
            single_channel[matches] = class_idx

        return single_channel

    def decode_single_channel_to_color_mask(self, single_channel_mask, color_map):
        """
        Convert a single-channel label mask (H, W) to a color-coded mask (H, W, 3).
        """
        h, w = single_channel_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, rgb in color_map.items():
            color_mask[single_channel_mask == class_idx] = rgb

        return color_mask

    # Image augmentations
    def apply_augmentations(self, is_train):
        # 2-class freespace colormap
        COLOR_MAP = {
            0: (61, 93, 255),      # non_drivable
            1: (0, 255, 220)       # drivable
        }

        if is_train:
            # Augmenting Data for training
            augTrain = Augmentations(is_train=True, data_type='SEGMENTATION')

            # Convert gt[0] (vis color mask) to single-channel indices for augmentation
            self.gt[0] = self.encode_color_mask_to_single_channel(self.gt[0], COLOR_MAP)

            augTrain.setData(self.image, self.gt)
            self.image, self.augmented = augTrain.applyTransformSeg(
                image=self.image, ground_truth=self.gt
            )

            # Convert back to color masks for visualization
            self.gt[0] = self.decode_single_channel_to_color_mask(self.gt[0], COLOR_MAP)
            self.augmented[0] = self.decode_single_channel_to_color_mask(self.augmented[0], COLOR_MAP)

            # Ground truth probabilities for each class in separate channels (H,W,2)
            # augmented[1] = non_drivable mask, augmented[2] = drivable mask
            self.gt_fused = np.stack((self.augmented[1], self.augmented[2]), axis=2)

        else:
            # Augmenting Data for testing/validation
            augVal = Augmentations(is_train=False, data_type='SEGMENTATION')

            # Convert gt_val[0] (vis color mask) to single-channel indices for augmentation
            self.gt_val[0] = self.encode_color_mask_to_single_channel(self.gt_val[0], COLOR_MAP)

            augVal.setData(self.image_val, self.gt_val)
            self.image_val, self.augmented_val = augVal.applyTransformSeg(
                image=self.image_val, ground_truth=self.gt_val
            )

            # Convert back to color masks for visualization
            self.gt_val[0] = self.decode_single_channel_to_color_mask(self.gt_val[0], COLOR_MAP)
            self.augmented_val[0] = self.decode_single_channel_to_color_mask(self.augmented_val[0], COLOR_MAP)

            # Ground truth probabilities for each class in separate channels (H,W,2)
            self.gt_val_fused = np.stack((self.augmented_val[1], self.augmented_val[2]), axis=2)

    # Load Data
    def load_data(self, is_train):
        self.load_image_tensor(is_train)
        self.load_gt_tensor(is_train)

        if is_train:
            # class_weights expected length 2
            self.class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)

    # Run Model (train only; validation uses calc_IoU_val)
    def run_model(self):
        # CrossEntropy expects:
        #   prediction: (N,C,H,W) logits
        #   target: (N,H,W) int64 class indices
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        self.contour_loss = nn.CrossEntropyLoss()
        
        _, self.contour_prediction = self.model(self.image_tensor)
        
        # Contour Loss
        # contour_prediction: (B, 37, 46, 1) -> (B, 37, 46)
        # We need to reshape for CrossEntropy.
        # Target: (B, 37) with values 0-45 (bin location)
        # Prediction needs to be (N, C, ...) where C is classes (46 bins).
        # So we want (B, 46, 37).
        
        contour_pred_reshaped = self.contour_prediction.squeeze(-1).permute(0, 2, 1) # (B, 32, 37)
        
        contour_loss_val = self.contour_loss(contour_pred_reshaped, self.gt_contour_tensor)
        
        # Store for logging
        self.contour_loss_val = contour_loss_val.item()
        
        self.calc_loss = contour_loss_val

    # Loss Backward Pass
    def loss_backward(self):
        self.calc_loss.backward()

    # Get loss value
    def get_loss(self):
        return self.calc_loss.item()
        
    def get_contour_loss(self):
        return self.contour_loss_val

    # Run Optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()

    # Save predicted visualization
    def save_visualization(self, log_count):
        print('Saving Visualization')
        # contour_prediction: (1, 37, 46, 1) -> (37, 46)
        # We need indices (37,)
        
        contour_pred = self.contour_prediction.squeeze(0).cpu().detach() # (37, 46, 1) or (37, 46)?
        # Original output from model is (B, 37, 46, 1)
        # Squeeze 0 (batch) -> (37, 46, 1)
        
        contour_pred = contour_pred.squeeze(-1) # (37, 46)
        contour_indices = torch.argmax(contour_pred, dim=1).numpy() # (37,)

        # Pass one of the augmented images (or self.image) to make_visualization
        # self.image is numpy array (H, W, 3) ? No, it's PIL Image or Tensor?
        # In set_data: self.image = image (PIL)
        # In apply_augmentations: self.image is transformed?
        # Augmentations.applyTransformSeg returns (image, mask) as numpy arrays if I recall?
        # Let's check Augmentations... assume self.image is numpy (H, W, 3) after augmentation.
        
        # Actually in trainer:
        # self.image, self.augmented = augTrain.applyTransformSeg(...)
        # So self.image is the augmented image (numpy array).
        
        vis_predict = self.make_visualization(self.image, contour_indices)
        
        label = self.augmented[0]
        
        # Plotting
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(self.image)
        axs[0].set_title('Image', fontweight="bold")
        axs[1].imshow(label)
        axs[1].set_title('Ground Truth', fontweight="bold")
        axs[2].imshow(vis_predict)
        axs[2].set_title('Prediction', fontweight="bold")

        save_path = os.path.join(os.getcwd(), f"vis_{log_count}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

        self.writer.add_figure('predictions vs. actuals', fig, global_step=(log_count))
        plt.close(fig)

    # Load Image as Tensor
    def load_image_tensor(self, is_train):
        if is_train:
            image_tensor = self.image_loader(self.image)
            image_tensor = image_tensor.unsqueeze(0)
            self.image_tensor = image_tensor.to(self.device)
        else:
            image_val_tensor = self.image_loader(self.image_val)
            image_val_tensor = image_val_tensor.unsqueeze(0)
            self.image_val_tensor = image_val_tensor.to(self.device)

    # Load Ground Truth as Tensor (convert one-hot (H,W,2) -> indices (H,W))
    def load_gt_tensor(self, is_train):
        if is_train:
            # gt_fused: (H,W,2) float/binary
            gt_idx = np.argmax(self.gt_fused, axis=2).astype(np.int64)  # (H,W)
            self.gt_tensor = torch.from_numpy(gt_idx).unsqueeze(0).to(self.device)  # (1,H,W) long
        else:
            gt_idx = np.argmax(self.gt_val_fused, axis=2).astype(np.int64)
            self.gt_val_tensor = torch.from_numpy(gt_idx).unsqueeze(0).to(self.device)

        # Generate Radial Contour GT
        # We need the original label image (before resizing/augmentation? Or after?)
        # RadialContourDists expects image (H, W).
        # self.gt_fused is (H,W,2), we need single channel drivable mask?
        # Or use self.gt[0] if it's the color mask?
        # RadialContourDists uses "drivable_label = 1".
        
        # Let's use the single channel mask we derived (gt_idx).
        # gt_idx: (H, W). 0=non-drivable, 1=drivable.
        
        # If batch size > 1, this needs a loop or vectorized RadialContourDists.
        # Current code seems to handle batch size 1 (self.image is one image).
        # load_gt_tensor unsqueezes to (1,H,W).
        
        # Compute contour indices
        # RadialContourDists(image, num_rays=37, ray_slice_dist=10, max_search_dist=320)
        
        # Note: gt_idx is numpy array (H,W)
        contour_gen = RadialContourDists(gt_idx, num_rays=37, ray_slice_dist=10, max_search_dist=460)
        contour_indices = contour_gen.get_contour_dists() # List of ints
        
        # Convert to tensor (B, 37)
        contour_tensor = torch.tensor(contour_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        
        if is_train:
            self.gt_contour_tensor = contour_tensor
        else:
            self.gt_val_contour_tensor = contour_tensor

    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)

    # Calculate Contour Accuracy and MAE for validation
    def calc_contour_metrics(self):
        _, contour_output = self.model(self.image_val_tensor)   # (1, C, H, W) -> contour: (1, 37, 46, 1)
        
        # Squeeze to (1, 37, 46) -> argmax to (1, 37) -> squeeze to (37)
        pred = torch.argmax(contour_output.squeeze(-1), dim=2).squeeze(0) # (37)
        
        gt = self.gt_val_contour_tensor.squeeze(0) # (37)
        
        # Accuracy: Percentage of rays where pred == gt
        accuracy = (pred == gt).float().mean().item()
        
        # MAE: Mean Absolute Error of bin indices
        mae = torch.abs(pred - gt).float().mean().item()

        return accuracy, mae

    # Run Validation and calculate metrics
    def validate(self, image_val, gt_val):
        # Set Data
        self.set_val_data(image_val, gt_val)

        # Augmenting Image
        self.apply_augmentations(is_train=False)

        # Converting to tensor and loading
        self.load_data(is_train=False)

        # Calculate Contour Metrics
        accuracy, mae = self.calc_contour_metrics()

        return accuracy, mae

    # Visualize predicted result
    # Visualize predicted result
    def make_visualization(self, image, contour_indices):
        # Image is (H, W, 3) numpy array
        vis_image = image.copy()
        height = vis_image.shape[0]
        width = vis_image.shape[1]
    
        # Reconstruct angles Left (Pi) to Right (0)
        num_rays = len(contour_indices)
        angles = np.linspace(np.pi, 0, num_rays)
        
        start_r = height - 1
        start_c = width // 2
        
        ray_slice_dist = 10 
        
        for idx, angle in zip(contour_indices, angles):
            dist = idx * ray_slice_dist
            # Calculate coordinates
            end_r = int(start_r - dist * np.sin(angle))
            end_c = int(start_c + dist * np.cos(angle))
            
            if 0 <= end_r < height and 0 <= end_c < width:
                cv2.circle(vis_image, (end_c, end_r), 5, (0, 255, 0), thickness=-1)

        return vis_image

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')
