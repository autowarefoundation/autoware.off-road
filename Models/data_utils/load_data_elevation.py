#! /usr/bin/env python3

import pathlib
import numpy as np
from PIL import Image


DATASETS = ['rellis3d']


class LoadDataElevation:
    """Dataset loader for the ElevationNetwork.

    Expected directory layout (under root):
        <dataset>/
        ├── images/          *.png   — RGB frames (any resolution; resized by augmentations)
        ├── gt_masks/        *.png   — uint8 elevation bin index maps, full image resolution
        │                             pixel value k → bin centre −0.5 + k×0.05 + 0.025 m
        └── camera_params/   *.npy   — float32 (12,) flattened [R|t] 3×4 projection matrix

    Files are matched by sorted filename stem.  Every 10th sample is held out
    for validation (90 / 10 split), consistent with other loaders in this repo.

    Args:
        labels_filepath        (str): Path to gt_masks/ directory.
        images_filepath        (str): Path to images/ directory.
        camera_params_filepath (str): Path to camera_params/ directory.
        dataset                (str): Must be one of DATASETS.
    """

    # Elevation bin constants — must match elevation_head.py
    H_MIN  = -0.50   # metres
    H_STEP =  0.05   # metres  (5 cm)
    NUM_BINS = 40

    def __init__(self, labels_filepath, images_filepath, camera_params_filepath, dataset: str):

        self.dataset = dataset
        if self.dataset not in DATASETS:
            raise ValueError(f'Dataset "{dataset}" not in {DATASETS}')

        self.labels        = sorted(pathlib.Path(labels_filepath).glob('*.png'))
        self.images        = sorted(pathlib.Path(images_filepath).glob('*.png'))
        self.camera_params = sorted(pathlib.Path(camera_params_filepath).glob('*.npy'))

        if len(self.images) != len(self.labels):
            raise ValueError('Mismatch between number of images and gt_masks')
        if len(self.images) != len(self.camera_params):
            raise ValueError('Mismatch between number of images and camera_params')
        if len(self.images) == 0:
            raise ValueError('No samples found — check the root paths')

        self.train_images, self.val_images           = [], []
        self.train_labels, self.val_labels           = [], []
        self.train_cam_params, self.val_cam_params   = [], []
        self.num_train_samples = 0
        self.num_val_samples   = 0

        for i, (img, lbl, cam) in enumerate(
                zip(self.images, self.labels, self.camera_params)):
            if (i + 1) % 10 == 0:
                self.val_images.append(str(img))
                self.val_labels.append(str(lbl))
                self.val_cam_params.append(str(cam))
                self.num_val_samples += 1
            else:
                self.train_images.append(str(img))
                self.train_labels.append(str(lbl))
                self.train_cam_params.append(str(cam))
                self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples

    def _load(self, img_path, lbl_path, cam_path):
        image        = np.array(Image.open(img_path).convert('RGB'))    # (H, W, 3) uint8
        label        = np.array(Image.open(lbl_path).convert('L'))      # (H, W)    uint8  0–39
        camera_params = np.load(cam_path).astype(np.float32)             # (12,)
        return image, label, camera_params

    def getItemTrain(self, index):
        return self._load(
            self.train_images[index],
            self.train_labels[index],
            self.train_cam_params[index],
        )

    def getItemVal(self, index):
        return self._load(
            self.val_images[index],
            self.val_labels[index],
            self.val_cam_params[index],
        )

    def getItemTrainPath(self, index):
        return self.train_images[index], self.train_labels[index], self.train_cam_params[index]

    def getItemValPath(self, index):
        return self.val_images[index], self.val_labels[index], self.val_cam_params[index]

    @staticmethod
    def bin_to_metres(label_uint8: np.ndarray) -> np.ndarray:
        """Convert a bin-index label map to expected elevation in metres (bin centres)."""
        return label_uint8.astype(np.float32) * LoadDataElevation.H_STEP \
               + LoadDataElevation.H_MIN + LoadDataElevation.H_STEP / 2.0

    @staticmethod
    def metres_to_bin(height_map: np.ndarray) -> np.ndarray:
        """Convert a float height map (metres) to uint8 bin indices 0–39."""
        bins = np.floor(
            (height_map - LoadDataElevation.H_MIN) / LoadDataElevation.H_STEP
        ).astype(np.int32)
        return np.clip(bins, 0, LoadDataElevation.NUM_BINS - 1).astype(np.uint8)
