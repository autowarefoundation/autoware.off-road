import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from datamodule.utils import get_mean_std, build_segformer_train_augs, read_label_mapping
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L

class GooseDataset(Dataset):
    def __init__(self, root_dir, split="train", label_mapping_name=None, crop_size=(512,512), scale_base=(2048,512), ratio_range=(0.5,2.0), mean=(123.675,116.28,103.53), std=(58.395,57.12,57.375), fine_to_superclass=None, superclass_colors=None):
        img_pattern = os.path.join(root_dir, "images", split, "**", "*_vis.png")
        #mask_pattern = os.path.join(root_dir, "labels", split, "**", "*_color.png")
        self.img_paths = sorted(glob(img_pattern, recursive=True))
        self.img_paths, self.mask_paths = self.check_image_mask_path()
        self.fine_to_superclass = np.array(fine_to_superclass,dtype=np.int64)
        self.superclass_colors = np.array(superclass_colors,dtype=np.int64)
        self.transform = build_segformer_train_augs(crop_size, scale_base, ratio_range, mean, std) if split == "train" else A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ])
        if label_mapping_name is not None:
            self.label_mapping_path = os.path.join(root_dir, label_mapping_name)
            self.label_id_to_name, self.rgb_to_label_id = read_label_mapping(self.label_mapping_path)
        else:
            self.label_id_to_name = None
            self.rgb_to_label_id = None

    def __len__(self):
        return len(self.img_paths)

    def remap_label(self, label):
        if self.fine_to_superclass is not None:
            return self.fine_to_superclass[label]
        return label

    def colorize_superclass_mask(self, mask):
        h, w = mask.shape
        colorized_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(self.superclass_colors.shape[0]):
            colorized_mask[mask == i] = self.superclass_colors[i]
        return colorized_mask

    def check_image_mask_path(self):
        image_paths, mask_paths = [], []
        for image_path in self.img_paths:
            mask_path = image_path.replace("images", "labels").replace("_windshield_vis.png", "_color.png")
            if not os.path.exists(mask_path):
                #print(f"Mask path not found for {image_path}")
                continue
            image_paths.append(image_path)
            mask_paths.append(mask_path)
        return image_paths, mask_paths

    def rgb_mask_to_label(self, mask_rgb):
        """
        Convert an RGB mask (H, W, 3) to a label mask (H, W) using self.rgb_to_label_id.
        """
        if self.rgb_to_label_id is None:
            raise ValueError("rgb_to_label_id mapping is not set.")
        h, w, _ = mask_rgb.shape
        label_mask = np.zeros((h, w), dtype=np.int64)
        for rgb, label in self.rgb_to_label_id.items():
            matches = np.all(mask_rgb == rgb, axis=-1)
            label_mask[matches] = label
        return label_mask

    def __getitem__(self, idx):
        img_fp = self.img_paths[idx]
        mask_fp = self.mask_paths[idx]
        img = Image.open(img_fp).convert('RGB')
        mask = Image.open(mask_fp).convert('RGB')


        #save original image and mask
        # cv2.imwrite("original_image.png", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        # cv2.imwrite("original_mask.png", cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR))

        img = np.array(img)
        mask = np.array(mask)
        if self.rgb_to_label_id is not None:
            mask = self.rgb_mask_to_label(mask)
        if self.fine_to_superclass is not None:
            mask = self.remap_label(mask)
        if self.transform:
            out = self.transform(image=img, mask=mask)

        #save transformed image and mask
        # img_t = out["image"]                    # torch.Size([3, H, W]), float32
        # img_np = img_t.permute(1, 2, 0).cpu().numpy()     # -> HWC, still float32
        # img_np = (img_np * 255).clip(0, 255).astype("uint8")   # -> uint8 0-255
        # cv2.imwrite("transformed_image.png",
        #             cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        # Convert tensor to numpy array for saving

        image = out["image"].float()                                # torch.float32
        mask  = out["mask"].squeeze(0).long() 
        # if self.rgb_to_label_id is not None:
        #     mask = self.rgb_mask_to_label(mask.numpy())
        return image, mask


class GooseDataModule(L.LightningDataModule):
    def __init__(self, root, label_mapping_path, batch_size=4, num_workers=4, num_classes=12, img_size=(448,448), scale_base=(2048,512), ratio_range=(0.5,2.0), mean=(123.675,116.28,103.53), std=(58.395,57.12,57.375), fine_to_superclass=None, superclass_colors=None):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = GooseDataset(root, "train", label_mapping_path, img_size, scale_base, ratio_range, mean, std, fine_to_superclass, superclass_colors)
        self.val_ds   = GooseDataset(root, "val",   label_mapping_path, img_size, scale_base, ratio_range, mean, std, fine_to_superclass, superclass_colors)
        self.test_ds  = GooseDataset(root, "test",  label_mapping_path, img_size, scale_base, ratio_range, mean, std, fine_to_superclass, superclass_colors)   
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True,   num_workers=self.num_workers,
                          pin_memory=True, drop_last=True) 

    def val_dataloader(self):
        return DataLoader(self.val_ds,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Goose Dataset Visualization')
    parser.add_argument('--root_dir', type=str, 
                       default="/home/autokarthik/autoware.off-road/goose-dataset",
                       help='Root directory of the dataset')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to visualize')
    parser.add_argument('--output', type=str, default='dataset_visualization.png',
                       help='Output filename for visualization')
    parser.add_argument('--index', type=int, default=0,
                       help='Index of the sample to visualize')
    
    args = parser.parse_args()

    fine_to_superclass = [7, 8, 1, 1, 6, 0, 6, 5, 7, 5, 8, 5, 4, 4, 11, 4, 0, 0, 0, 8, 4, 5, 5, 1, 1, 8, 5, 0, 0, 7, 0, 1, 11, 10, 4, 4, 4, 4, 3, 3, 6, 3, 3, 3, 3, 6, 8, 8, 8, 4, 0, 0, 0, 2, 9, 3, 7, 4, 3, 0, 6, 6, 0, 4]
    superclass_colors = [[128, 150, 147], [255, 215, 255], [183, 121, 164], [15, 51, 181], [153, 121, 215], [122, 73, 0], [255, 74, 70], [120, 141, 120], [194, 255, 153], [122, 135, 161], [0, 0, 53], [221, 239, 255]]
    
    dataset = GooseDataset(root_dir=args.root_dir, split=args.split, label_mapping_name="goose_label_mapping.csv", crop_size=(512,512), fine_to_superclass=fine_to_superclass, superclass_colors=superclass_colors)
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(y.unique(), y.min(), y.max())

    mean, std = get_mean_std(dataset)

    print(f"mean: {mean}")
    print(f"std: {std}")

    # img, mask = dataset[args.index]["image"], dataset[args.index]["mask"]
    # img = img.permute(1, 2, 0).numpy()
    # mask = mask.numpy()
    
    # # Create 1x2 subplot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # # Plot original image
    # ax1.imshow(img)
    # ax1.set_title('Original Image')
    # ax1.axis('off')
    
    # # Plot mask
    # ax2.imshow(mask)
    # ax2.set_title('Mask')
    # ax2.axis('off')
    
    # plt.tight_layout()
    # plt.savefig(args.output, dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"Visualization saved as '{args.output}'")


