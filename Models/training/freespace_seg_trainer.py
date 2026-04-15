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


class FreespaceSegTrainer():
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
        self.class_weights_tensor = 0
        self.loss = 0
        self.prediction = 0
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
    def log_loss(self, log_count, loss=None):
        if loss is None:
            loss = self.get_loss()
        print('Logging Training Loss', log_count, loss)
        self.writer.add_scalar("Loss/train", loss, (log_count))

    # Logging Validation mIoU Score (2 classes)
    def log_IoU(self, mIoU_full, mIoU_bg, mIoU_fg, log_count):
        print('Logging Validation')
        self.writer.add_scalars("Val/IoU_Classes", {
            'mIoU_non_drivable': mIoU_bg,
            'mIoU_drivable': mIoU_fg,
        }, (log_count))
        self.writer.add_scalar("Val/IoU", mIoU_full, (log_count))

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
        
        self.prediction, _ = self.model(self.image_tensor)
        
        # Segmentation Loss
        self.calc_loss = self.loss(self.prediction, self.gt_tensor)

    # Loss Backward Pass
    def loss_backward(self):
        self.calc_loss.backward()

    # Get loss value
    def get_loss(self):
        return self.calc_loss.item()

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
        self.prediction_vis = self.prediction.squeeze(0).cpu().detach()
        self.prediction_vis = self.prediction_vis.permute(1, 2, 0)  # (H,W,C)

        vis_predict = self.make_visualization()
        label = self.augmented[0]
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

    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)

    # Calculate IoU score for validation (2 classes)
    def calc_IoU_val(self):
        output_val, _ = self.model(self.image_val_tensor)   # (1, C, H, W)
        pred = torch.argmax(output_val, dim=1)           # (1, H, W)

        one_hot = torch.nn.functional.one_hot(pred, num_classes=2).squeeze(0)  # (H, W, 2)
        one_hot = one_hot.cpu().detach().numpy()

        iou_score_full = self.IoU(one_hot, self.gt_val_fused)
        iou_score_bg = self.IoU(one_hot[..., 0], self.gt_val_fused[..., 0])
        iou_score_fg = self.IoU(one_hot[..., 1], self.gt_val_fused[..., 1])

        return iou_score_full, iou_score_bg, iou_score_fg

    # IoU calculation
    def IoU(self, output, label):
        intersection = np.logical_and(label, output)
        union = np.logical_or(label, output)
        iou_score = (np.sum(intersection) + 1) / float(np.sum(union) + 1)
        return iou_score

    # Run Validation and calculate metrics
    def validate(self, image_val, gt_val):
        # Set Data
        self.set_val_data(image_val, gt_val)

        # Augmenting Image
        self.apply_augmentations(is_train=False)

        # Converting to tensor and loading
        self.load_data(is_train=False)

        # Calculate IoU score
        iou_score_full, iou_score_bg, iou_score_fg = self.calc_IoU_val()

        return iou_score_full, iou_score_bg, iou_score_fg

    # Visualize predicted result
    def make_visualization(self):
        shape = self.prediction_vis.shape
        _, output = torch.max(self.prediction_vis, dim=2)

        row = shape[0]
        col = shape[1]
        vis_predict = Image.new(mode="RGB", size=(col, row))
        vx = vis_predict.load()

        non_drivable_colour = (61, 93, 255)
        drivable_colour = (0, 255, 220)

        # Extracting predicted classes and assigning to colourmap
        for x in range(row):
            for y in range(col):
                if output[x, y].item() == 0:
                    vx[y, x] = non_drivable_colour
                elif output[x, y].item() == 1:
                    vx[y, x] = drivable_colour

        return vis_predict

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')