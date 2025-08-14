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
        self.cfg = cfg 
        dataset_dir = Path(cfg.dataset_dir)

        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"dataset directory not found: {dataset_dir}"
            )

        image_paths = glob(str(dataset_dir / cfg.label_config.src_image_pattern), recursive=True)
        mask_paths = glob(str(dataset_dir / cfg.label_config.src_gt_mask_pattern), recursive=True)

        exclude_patterns = cfg.label_config.get("exclude_patterns", [])
        if exclude_patterns:
            mask_paths = [
                p for p in mask_paths
                if not any(fnmatch(str(Path(p).relative_to(dataset_dir)), pat) for pat in exclude_patterns)
            ]
            image_paths = [
                p for p in image_paths
                if not any(fnmatch(str(Path(p).relative_to(dataset_dir)), pat) for pat in exclude_patterns)
            ]

        filename_mapping = cfg.label_config.get("filename_mapping", {})

        image_stem_to_path = {self.apply_mapping(Path(p).stem): p for p in image_paths}
        mask_stem_to_path = {self.apply_mapping(Path(p).stem): p for p in mask_paths}

        common_stems = set(image_stem_to_path.keys()) & set(mask_stem_to_path.keys())

        self.dataset = cfg.label_config.dataset
        self.image_paths = sorted(OrderedDict.fromkeys([image_stem_to_path[s] for s in common_stems]))
        self.mask_label_paths = sorted(OrderedDict.fromkeys([mask_stem_to_path[s] for s in common_stems]))

        self.dst_image_dir = dataset_dir / cfg.label_config.dst_image_dir
        self.dst_gt_mask_dir = dataset_dir / cfg.label_config.dst_gt_mask_dir

        self.label_map = self._parse_label_map(cfg.label_config.label_to_id_mapping)
        self.id_to_color_map = cfg.color_map.id_to_color_mapping
        self.overwrite = cfg.label_config.overwrite_existing
        self.color_thresh = cfg.label_config.color_thresh

    def _parse_label_map(self, raw_map):
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

    def apply_mapping(self, stem):
        filename_mapping = self.cfg.label_config.get("filename_mapping", {})

        for src, tgt in filename_mapping.items():
            if src in stem:
                stem = stem.replace(src, tgt)
        return stem

    def build_dst_image_copy(self):
        for i, src_path in enumerate(self.image_paths):
            remapped_stem = self.apply_mapping(Path(src_path).stem)
            dst_filename = remapped_stem + ".png"
            dst_path = self.dst_image_dir / dst_filename
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            prefix = f"[{self.dataset}]"

            if not self.overwrite and dst_path.is_file():
                print(f"{prefix}: copy_image {i+1}/{len(self.image_paths)} - {dst_path.name} (skipped)")
                continue

            img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"{prefix}: copy_image {i+1}/{len(self.image_paths)} - Failed to read: {src_path}")
                continue

            cv2.imwrite(str(dst_path), img)
            print(f"{prefix}: copy_image {i+1}/{len(self.image_paths)} - {dst_path.name} (saved)")

    def build_dst_mask_color(self):
        for i, src_path in enumerate(self.mask_label_paths):
            remapped_stem = self.apply_mapping(Path(src_path).stem)
            dst_filename = remapped_stem + ".png"
            dst_path = self.dst_gt_mask_dir / dst_filename
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            prefix = f"[{self.dataset}]"

            mask = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"{prefix}: mask_color {i+1}/{len(self.mask_label_paths)} - Failed to read: {src_path}")
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

            if self.overwrite or not dst_path.is_file():
                cv2.imwrite(str(dst_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
                print(f"{prefix}: mask_color {i+1}/{len(self.mask_label_paths)} - {dst_path.name} (saved)")
            else:
                print(f"{prefix}: mask_color {i+1}/{len(self.mask_label_paths)} - {dst_path.name} (skipped)")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dataset = PrepareDataset(cfg)
    dataset.build_dst_image_copy()
    dataset.build_dst_mask_color()

if __name__ == "__main__":
    main()