#! /usr/bin/env python3

import pathlib
import numpy as np
from PIL import Image


DATASETS = ['rellis3d']


class LoadDataElevation:
    """Dataset loader for the ElevationNetwork.

    Expected directory layout (under root):
        <dataset>/
        ├── images/                  *.png        — RGB frames (any resolution; resized by augmentations)
        ├── gt_elevations/           *.png        — uint8 grayscale 96–255 (scaled bin indices)
        │                                           pixel value p → bin index round((p - 96) * 39 / 159)
        └── camera_params/           camera_params.npy — float32 (12,) flattened [R|t] 3×4 projection
                                                          matrix, shared by all frames

    Files are matched by sorted filename stem.  Every 10th sample is held out
    for validation (90 / 10 split), consistent with other loaders in this repo.

    Args:
        labels_filepath        (str): Path to gt_elevations/ directory.
        images_filepath        (str): Path to images/ directory.
        camera_params_filepath (str): Path to camera_params/ directory (contains camera_params.npy).
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

        self.labels = sorted(pathlib.Path(labels_filepath).glob('*.png'))
        self.images = sorted(pathlib.Path(images_filepath).glob('*.png'))

        # Single shared camera params file
        cam_file = pathlib.Path(camera_params_filepath) / 'camera_params.npy'
        if not cam_file.exists():
            raise FileNotFoundError(f'camera_params.npy not found in {camera_params_filepath}')
        self.shared_cam_params = np.load(str(cam_file)).astype(np.float32)  # (12,)

        if len(self.images) != len(self.labels):
            raise ValueError('Mismatch between number of images and gt_elevations')
        if len(self.images) == 0:
            raise ValueError('No samples found — check the root paths')

        self.train_images, self.val_images = [], []
        self.train_labels, self.val_labels = [], []
        self.num_train_samples = 0
        self.num_val_samples   = 0

        for i, (img, lbl) in enumerate(zip(self.images, self.labels)):
            if (i + 1) % 10 == 0:
                self.val_images.append(str(img))
                self.val_labels.append(str(lbl))
                self.num_val_samples += 1
            else:
                self.train_images.append(str(img))
                self.train_labels.append(str(lbl))
                self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples

    def _load(self, img_path, lbl_path):
        image     = np.array(Image.open(img_path).convert('RGB'))  # (H, W, 3) uint8
        label_raw = np.array(Image.open(lbl_path).convert('L'))    # (H, W)    uint8  0–255
        # Rescale 0–255 back to bin indices 0–(NUM_BINS-1)
        label     = np.round((label_raw.astype(np.float32) - 96.0) * (LoadDataElevation.NUM_BINS - 1) / 159.0).astype(np.uint8)
        return image, label, self.shared_cam_params

    def getItemTrain(self, index):
        return self._load(self.train_images[index], self.train_labels[index])

    def getItemVal(self, index):
        return self._load(self.val_images[index], self.val_labels[index])

    def getItemTrainPath(self, index):
        return self.train_images[index], self.train_labels[index]

    def getItemValPath(self, index):
        return self.val_images[index], self.val_labels[index]

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
