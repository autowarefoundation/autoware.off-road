import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
import yaml

class Rellis3DDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, label_map=None):
        img = os.path.join(root_dir, "Rellis_3D_pylon_camera_node", "**", "**", "**", "*.jpg")
        mask_label = os.path.join(root_dir, "Rellis_3D_pylon_camera_node_label_color", "**", "**","**", "*.png")
        mask_id = os.path.join(root_dir, "Rellis_3D_pylon_camera_node_label_id", "**", "**", "**", "*.png")
        self.img_paths = sorted(glob(img, recursive=True))
        self.mask_label_paths = sorted(glob(mask_pattern, recursive=True))
        self.mask_id_paths = sorted(glob(mask_pattern, recursive=True))
        self.transform = transform
        self.label_map = label_map  # e.g., {0:0, 1:0, 2:1}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_fp = self.img_paths[idx]
        mask_fp = self.mask_paths[idx]
        img = cv2.imread(img_fp, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        # Apply label mapping if provided
        if self.label_map:
            remapped_mask = np.copy(mask)
            for old_label, new_label in self.label_map.items():
                remapped_mask[mask == old_label] = new_label
            mask = remapped_mask

        if self.transform:
            img = self.transform(img)

        return {"image": torch.from_numpy(img).permute(2, 0, 1), "mask": torch.from_numpy(mask)}

def load_label_map(config_path):
    """Load label remapping dictionary from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get("label_map", {})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Goose Dataset Visualization')
    parser.add_argument('--root_dir', type=str,
                        default="/home/autokarthik/autoware.off-road/goose-dataset",
                        help='Root directory of the dataset')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--config', type=str, default='config/class_merge.yaml',
                        help='Path to YAML config file for label mapping')
    parser.add_argument('--output', type=str, default='dataset_visualization.png',
                        help='Output filename for visualization')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of the sample to visualize')
    
    args = parser.parse_args()

    label_map = None
    if args.config and os.path.exists(args.config):
        label_map = load_label_map(args.config)
        print(f"Loaded label map: {label_map}")
    else:
        print("No label map config found or provided. Using original labels.")

    dataset = Rellis3DDataset(root_dir=args.root_dir, split=args.split, label_map=label_map)
    print(f"Dataset size: {len(dataset)}")
    
    mean, std = get_mean_std(dataset)

    print(f"mean: {mean}")
    print(f"std: {std}")

    img, mask = dataset[args.index]["image"], dataset[args.index]["mask"]
    img = img.permute(1, 2, 0).numpy()
    mask = mask.numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(mask)
    ax2.set_title('Merged Mask')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as '{args.output}'")
