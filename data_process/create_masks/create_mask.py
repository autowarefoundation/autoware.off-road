import os
from glob import glob
import cv2
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from collections import OrderedDict


class PrepareDataset:
    def __init__(self, cfg):
        dataset_dir = Path(cfg.dataset_dir)

        mask_paths = glob(str(dataset_dir / cfg.dataset.src_mask_label_pattern), recursive=True)
        self.mask_label_paths = sorted(OrderedDict.fromkeys(mask_paths))

        self.dest_mask_id_dir = dataset_dir / cfg.dataset.dst_mask_id_root
        self.dest_mask_color_dir = dataset_dir / cfg.dataset.dst_mask_color_root

        self.keep_levels = cfg.dataset.keep_subdir_levels
        self.label_map = cfg.dataset.label_to_id_mapping
        self.id_to_color_map = cfg.color_map.id_to_color_mapping

        self.overwrite = cfg.dataset.overwrite_existing

    def get_relative_subpath(self, input_path, keep_levels=2):
        # Return the last `keep_levels` components of the input path.
        input_path = Path(input_path)
        return Path(*input_path.parts[-keep_levels:])

    def remap_mask_id(self, idx):
        # Load image (grayscale or color as-is)
        mask = cv2.imread(self.mask_label_paths[idx], cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read image at {self.mask_label_paths[idx]}")

        is_color = len(mask.shape) == 3 and mask.shape[2] == 3
        remapped = np.zeros(mask.shape[:2], dtype=np.uint8)

        for old_label, new_id in self.label_map.items():
            if is_color:
                if not isinstance(old_label, (list, tuple)) or len(old_label) != 3:
                    raise ValueError(f"Expected RGB list/tuple for color mask, got: {old_label}")
                rgb = tuple(old_label)
                match = np.all(mask == rgb, axis=-1)
            else:
                if not isinstance(old_label, int):
                    raise ValueError(f"Expected int for grayscale mask, got: {old_label}")
                match = mask == old_label

            remapped[match] = new_id

        return remapped

    def remap_mask_color(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, rgb in self.id_to_color_map.items():
            assert len(rgb) == 3, f"Invalid RGB for class {class_id}: {rgb}"
            color_mask[mask == class_id] = rgb
        return color_mask

    def build_dest_mask_id(self):
        for i, src_label_path in enumerate(self.mask_label_paths):
            remapped_mask = self.remap_mask_id(i)
            sub_path = self.get_relative_subpath(src_label_path, self.keep_levels)
            save_path = self.dest_mask_id_dir / sub_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if self.overwrite or not save_path.is_file():
                cv2.imwrite(str(save_path), remapped_mask)
                print(f"[mask_id] {i + 1}/{len(self.mask_label_paths)} - {sub_path} (saved)")
            else:
                print(f"[mask_id] {i + 1}/{len(self.mask_label_paths)} - {sub_path} (skipped, exists)")

    def build_dest_mask_color(self):
        for i, src_label_path in enumerate(self.mask_label_paths):
            sub_path = self.get_relative_subpath(src_label_path, self.keep_levels)
            remapped_id_path = self.dest_mask_id_dir / sub_path

            if not remapped_id_path.is_file():
                print(f"Skipping: missing ID mask at {remapped_id_path}")
                continue

            color_mask = self.remap_mask_color(str(remapped_id_path))
            color_mask_save_path = self.dest_mask_color_dir / sub_path
            color_mask_save_path.parent.mkdir(parents=True, exist_ok=True)

            if self.overwrite or not color_mask_save_path.is_file():
                cv2.imwrite(str(color_mask_save_path), color_mask)
                print(f"[mask_color] {i + 1}/{len(self.mask_label_paths)} - {sub_path} (saved)")
            else:
                print(f"[mask_color] {i + 1}/{len(self.mask_label_paths)} - {sub_path} (skipped, exists)")



@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dataset = PrepareDataset(cfg)
    dataset.build_dest_mask_id()
    dataset.build_dest_mask_color()


if __name__ == "__main__":
    main()
