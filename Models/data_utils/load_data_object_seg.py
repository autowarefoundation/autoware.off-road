#! /usr/bin/env python3

import pathlib
from PIL import Image
import numpy as np


# objectseg class color map (RGB) — matches objectseg.yaml
COLOR_MAP = {
    0: (61,  93, 255),   # background
    1: (255,  28, 145),  # man-made_structures
    2: (255,  61,  61),  # vulnerable_living
    3: (255, 190,  61),  # vehicle
    4: (  0, 100,   0),  # natural_obstacles
}
NUM_CLASSES = len(COLOR_MAP)

DATASETS = ['CARLA_Mining']


class LoadDataObjectSeg:
    def __init__(self, labels_filepath, images_filepath, dataset: str):

        self.dataset = dataset
        if self.dataset not in DATASETS:
            raise ValueError('Dataset type is not correctly specified')

        self.labels = sorted(
            [f for f in pathlib.Path(labels_filepath).glob('*.png')])
        self.images = sorted(
            [f for f in pathlib.Path(images_filepath).glob('*.png')])

        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        if self.num_images != self.num_labels:
            raise ValueError(
                'Number of images and ground truth labels are mismatched')

        if self.num_images == 0:
            raise ValueError('No images found - check the root path')

        if self.num_labels == 0:
            raise ValueError('No ground truth masks found - check the root path')

        self.train_images = []
        self.train_labels = []
        self.val_images   = []
        self.val_labels   = []

        self.num_train_samples = 0
        self.num_val_samples   = 0

        for count in range(self.num_images):
            if (count + 1) % 10 == 0:
                self.val_images.append(str(self.images[count]))
                self.val_labels.append(str(self.labels[count]))
                self.num_val_samples += 1
            else:
                self.train_images.append(str(self.images[count]))
                self.train_labels.append(str(self.labels[count]))
                self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples

    def createGroundTruth(self, input_label):
        input_np = np.array(input_label.convert('RGB'))
        row, col, _ = input_np.shape
        num_pixels  = row * col

        vis = input_np.astype(np.uint8)

        # Build per-class binary masks and class weights
        masks        = []
        class_weights = []
        epsilon = 5120

        for class_id in range(NUM_CLASSES):
            rgb  = COLOR_MAP[class_id]
            mask = np.all(input_np == rgb, axis=-1).astype(np.uint8) * 255
            masks.append(mask)
            class_weights.append(num_pixels / (np.sum(mask > 0) + epsilon))

        # gt[0] = vis color image, gt[1..N] = per-class binary masks
        ground_truth = [vis] + masks

        return ground_truth, class_weights

    def getItemTrain(self, index):
        image = Image.open(self.train_images[index]).convert('RGB')
        label = Image.open(self.train_labels[index])
        gt, class_weights = self.createGroundTruth(label)
        return np.array(image), gt, class_weights

    def getItemTrainPath(self, index):
        return self.train_images[index], self.train_labels[index]

    def getItemVal(self, index):
        image = Image.open(self.val_images[index]).convert('RGB')
        label = Image.open(self.val_labels[index])
        gt, class_weights = self.createGroundTruth(label)
        return np.array(image), gt, class_weights

    def getItemValPath(self, index):
        return self.val_images[index], self.val_labels[index]
