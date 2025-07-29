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

    def build_dst_mask_color(self):
        for i, src_label_path in enumerate(self.mask_label_paths):
            sub_path = self.get_relative_subpath(src_label_path, self.keep_levels)

            mask = cv2.imread(src_label_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"Warning: Failed to load mask at {src_label_path}")
                continue

            is_color = len(mask.shape) == 3 and mask.shape[2] in (3, 4)
            mask_rgb = None
            if is_color:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB if mask.shape[2] == 4 else cv2.COLOR_BGR2RGB)

            id_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

            for old_label, new_id in self.label_map.items():
                if is_color and isinstance(old_label, tuple):
                    diff = np.abs(mask_rgb - old_label)
                    match = np.all(diff <= self.color_thresh, axis=-1)
                elif not is_color and isinstance(old_label, int):
                    match = mask == old_label
                else:
                    continue
                id_mask[match] = new_id

            color_mask = np.zeros((*id_mask.shape, 3), dtype=np.uint8)
            for class_id, rgb in self.id_to_color_map.items():
                color_mask[id_mask == class_id] = rgb

            save_path = self.dst_mask_color_dir / sub_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if self.overwrite or not save_path.is_file():
                cv2.imwrite(str(save_path), color_mask)
                print(f"[mask_color] {i + 1}/{len(self.mask_label_paths)} - {sub_path} (saved)")
            else:
                print(f"[mask_color] {i + 1}/{len(self.mask_label_paths)} - {sub_path} (skipped, exists)")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dataset = PrepareDataset(cfg)
    dataset.build_dst_mask_color()


if __name__ == "__main__":
    main()
