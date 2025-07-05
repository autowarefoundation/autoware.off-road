import os
from glob import glob
import cv2
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


class PrepareDataset:
    def __init__(self, cfg):
        dataset_dir = Path(cfg.dataset_dir)

        self.mask_id_paths = sorted(glob(str(dataset_dir / cfg.dataset.mask_id), recursive=True))
        self.mask_color_paths = sorted(glob(str(dataset_dir / cfg.dataset.mask_color), recursive=True))

        self.new_mask_id_dir = dataset_dir / cfg.dataset.new_mask_id
        self.new_mask_color_dir = dataset_dir / cfg.dataset.new_mask_color

        self.id_map = cfg.dataset.id_map
        self.color_map = cfg.color_map.color_map

    def get_relative_subpath(self, full_path, root_dir, drop_levels=2):
        rel_path = Path(os.path.relpath(full_path, root_dir))
        return Path(*rel_path.parts[drop_levels:])

    def save_mask(self, image, save_dir, sub_path):
        full_path = save_dir / sub_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.is_file():
            cv2.imwrite(str(full_path), image)

    def remap_mask_id(self, idx):
        mask = cv2.imread(self.mask_id_paths[idx], cv2.IMREAD_GRAYSCALE)
        remapped = np.copy(mask)
        for old_id, new_id in self.id_map.items():
            remapped[mask == old_id] = new_id
        return remapped

    def remap_mask_color(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, rgb in self.color_map.items():
            assert len(rgb) == 3, f"Invalid RGB for class {class_id}: {rgb}"
            color_mask[mask == class_id] = rgb
        return color_mask  # Already in RGB if your map uses RGB

    def build_new_mask_id(self):
        for i, mask_path in enumerate(self.mask_id_paths):
            remapped = self.remap_mask_id(i)
            sub_path = self.get_relative_subpath(mask_path, self.new_mask_id_dir)
            self.save_mask(remapped, self.new_mask_id_dir, sub_path)
            print(f"[mask_id] {i + 1}/{len(self.mask_id_paths)}")

    def build_new_mask_color(self):
        for i, original_mask_path in enumerate(self.mask_id_paths):
            # Compute expected new mask ID path
            sub_path = self.get_relative_subpath(original_mask_path, self.new_mask_id_dir)
            new_mask_path = self.new_mask_id_dir / sub_path

            if not new_mask_path.is_file():
                print(f"Skipping: missing mask ID at {new_mask_path}")
                continue

            color_mask = self.remap_mask_color(str(new_mask_path))

            # Use original color path just for subdirectory structure
            color_sub_path = self.get_relative_subpath(self.mask_color_paths[i], self.new_mask_color_dir)
            full_img_path = self.new_mask_color_dir / color_sub_path

            full_img_path.parent.mkdir(parents=True, exist_ok=True)

            if not full_img_path.is_file():
                cv2.imwrite(str(full_img_path), color_mask)

            print(f"[mask_color] {i + 1}/{len(self.mask_id_paths)}")



@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    dataset = PrepareDataset(cfg)
    dataset.build_new_mask_id()
    dataset.build_new_mask_color()


if __name__ == "__main__":
    main()