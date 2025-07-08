import os
from glob import glob
import cv2
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from collections import OrderedDict
from fnmatch import fnmatch


class PrepareDataset:
    def __init__(self, cfg):
        dataset_dir = Path(cfg.dataset_dir)

        all_paths = glob(str(dataset_dir / cfg.label_config.src_mask_label_pattern), recursive=True)

        exclude_patterns = cfg.label_config.get("exclude_patterns", [])
        if exclude_patterns:
            filtered_paths = []
            for path in all_paths:
                rel_path = str(Path(path).relative_to(dataset_dir))
                if not any(fnmatch(rel_path, pattern) for pattern in exclude_patterns):
                    filtered_paths.append(path)
            mask_paths = filtered_paths
        else:
            mask_paths = all_paths

        self.mask_label_paths = sorted(OrderedDict.fromkeys(mask_paths))

        self.dst_mask_id_dir = dataset_dir / cfg.label_config.dst_mask_id_root
        self.dst_mask_color_dir = dataset_dir / cfg.label_config.dst_mask_color_root

        self.keep_levels = cfg.label_config.keep_subdir_levels
        self.label_map = self._parse_label_map(cfg.label_config.label_to_id_mapping)
        self.id_to_color_map = cfg.color_map.id_to_color_mapping
        
        self.overwrite = cfg.label_config.overwrite_existing
        self.color_thresh = cfg.label_config.color_thresh

    def _parse_label_map(self, raw_map):
        """Convert stringified RGB keys to tuples; keep integers as-is."""
        parsed = {}
        for key, value in raw_map.items():
            if isinstance(key, str) and key.startswith("[") and key.endswith("]"):
                try:
                    key = tuple(map(int, key.strip("[]").split(",")))
                except Exception:
                    raise ValueError(f"Invalid RGB key format: {key}")
            elif not isinstance(key, int):
                raise TypeError(f"Label key must be int or stringified RGB list, got: {type(key)}")
            parsed[key] = value
        return parsed


    def get_relative_subpath(self, input_path, keep_levels=2):
        input_path = Path(input_path)
        return Path(*input_path.parts[-keep_levels:])

    def remap_mask_id(self, idx):
        mask = cv2.imread(self.mask_label_paths[idx], cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read image at {self.mask_label_paths[idx]}")

        is_color = len(mask.shape) == 3 and mask.shape[2] in (3, 4)
        remapped = np.zeros(mask.shape[:2], dtype=np.uint8)

        if is_color:
            if mask.shape[2] == 4:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
            else:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        for old_label, new_id in self.label_map.items():
            if is_color and isinstance(old_label, tuple) and len(old_label) == 3:
                # Fuzzy RGB match using color_thresh
                diff = np.abs(mask_rgb - old_label)
                match = np.all(diff <= self.color_thresh, axis=-1)
            elif not is_color and isinstance(old_label, int):
                match = mask == old_label
            else:
                continue

            remapped[match] = new_id

        return remapped


    def remap_mask_color(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, rgb in self.id_to_color_map.items():
            assert len(rgb) == 3, f"Invalid RGB for class {class_id}: {rgb}"
            color_mask[mask == class_id] = rgb
        return color_mask

    def build_dst_mask_id(self):
        for i, src_label_path in enumerate(self.mask_label_paths):
            remapped_mask = self.remap_mask_id(i)
            sub_path = self.get_relative_subpath(src_label_path, self.keep_levels)
            save_path = self.dst_mask_id_dir / sub_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if self.overwrite or not save_path.is_file():
                cv2.imwrite(str(save_path), remapped_mask)
                print(f"[mask_id] {i + 1}/{len(self.mask_label_paths)} - {sub_path} (saved)")
            else:
                print(f"[mask_id] {i + 1}/{len(self.mask_label_paths)} - {sub_path} (skipped, exists)")

    def build_dst_mask_color(self):
        for i, src_label_path in enumerate(self.mask_label_paths):
            sub_path = self.get_relative_subpath(src_label_path, self.keep_levels)
            remapped_id_path = self.dst_mask_id_dir / sub_path

            if not remapped_id_path.is_file():
                print(f"Skipping: missing ID mask at {remapped_id_path}")
                continue

            color_mask = self.remap_mask_color(str(remapped_id_path))
            color_mask_save_path = self.dst_mask_color_dir / sub_path
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
    dataset.build_dst_mask_id()
    dataset.build_dst_mask_color()


if __name__ == "__main__":
    main()