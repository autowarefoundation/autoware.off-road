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
from utils import get_mean_std, build_segformer_train_augs
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GooseDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        img_pattern = os.path.join(root_dir, "images", split, "**", "*_vis.png")
        mask_pattern = os.path.join(root_dir, "labels", split, "**", "*_color.png")
        self.img_paths = sorted(glob(img_pattern, recursive=True))
        self.mask_paths = sorted(glob(mask_pattern, recursive=True))
        self.transform = build_segformer_train_augs() if split == "train" else A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=(123.675,116.28,103.53), std=(58.395,57.12,57.375), max_pixel_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_fp = self.img_paths[idx]
        mask_fp = self.mask_paths[idx]
        img = cv2.imread(img_fp, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE).astype(np.int64)
        if self.transform:
            img = self.transform(img)
        return {"image": torch.from_numpy(img).permute(2, 0, 1) # (C,H,W)
                , "mask": torch.from_numpy(mask).long()}


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
    
    dataset = GooseDataset(root_dir=args.root_dir, split=args.split)
    print(f"Dataset size: {len(dataset)}")

    #mean, std = get_mean_std(dataset)

    #print(f"mean: {mean}")
    #print(f"std: {std}")

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


