#! /usr/bin/env python3

from typing import Literal
import pathlib
from PIL import Image
import numpy as np


class LoadDataObjectSeg:
    def __init__(self, labels_filepath, images_filepath,
                 dataset: Literal['CaSSeD', 'Goose', 'OFFSED', 'ORFD', 'Rellis_3D', 'Yamaha_CMU']):

        self.dataset = dataset
        if self.dataset not in ['CaSSeD', 'Goose', 'OFFSED', 'ORFD', 'Rellis_3D', 'Yamaha_CMU']:
            raise ValueError('Dataset type is not correctly specified')

        self.labels = sorted(
            [f for f in pathlib.Path(labels_filepath).glob('*.png')])
        self.images = sorted(
            [f for f in pathlib.Path(images_filepath).glob('*.png')])

        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        if (self.num_images != self.num_labels):
            raise ValueError(
                'Number of images and ground truth labels are mismatched')

        if (self.num_images == 0):
            raise ValueError('No images found - check the root path')

        if (self.num_labels == 0):
            raise ValueError(
                'No ground truth masks found - check the root path')

        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        self.num_train_samples = 0
        self.num_val_samples = 0

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
        # Colourmaps for classes
        sky_colour = (61, 184, 255)
        background_objects_colour = (61, 93, 255)
        road_edge_delimiter_colour = (216, 255, 61)
        unlabelled_colour = (0, 0, 0)
        vulnerable_living_colour = (255, 61, 61)
        small_mobile_vehicle_colour = (255, 190, 61)
        large_mobile_vehicle_colour = (255, 116, 61)
        foreground_objects_colour = (255, 28, 145)
        road_colour = (0, 255, 220)

        background_colours = [background_objects_colour, road_edge_delimiter_colour, unlabelled_colour, sky_colour]
        foreground_colours = [vulnerable_living_colour, small_mobile_vehicle_colour, large_mobile_vehicle_colour, foreground_objects_colour]
        road_colours = [road_colour]

        # Convert input PIL image to NumPy array
        input_np = np.array(input_label)
        row, col, _ = input_np.shape
        num_pixels = row*col

        # Initialize visualization and masks
        vis = np.zeros_like(input_np, dtype=np.uint8)
        ground_truth_background = np.zeros((row, col), dtype=np.uint8)
        ground_truth_foreground = np.zeros((row, col), dtype=np.uint8)
        ground_truth_road = np.zeros((row, col), dtype=np.uint8)

        # Define boolean masks for classes
        background_mask = np.isin(input_np.reshape(-1, 3), background_colours).all(axis=1).reshape(row, col)
        foreground_mask = np.isin(input_np.reshape(-1, 3), foreground_colours).all(axis=1).reshape(row, col)
        road_mask = np.isin(input_np.reshape(-1, 3), road_colours).all(axis=1).reshape(row, col)

        # Apply masks
        vis[background_mask] = background_objects_colour
        vis[foreground_mask] = foreground_objects_colour
        vis[road_mask] = road_colour
        
        ground_truth_background[background_mask] = 255
        ground_truth_foreground[foreground_mask] = 255
        ground_truth_road[road_mask] = 255

        # Compute class frequencies
        background_class_freq = np.sum(background_mask)
        foreground_class_freq = np.sum(foreground_mask)
        road_class_freq = np.sum(road_mask)

        # Calculate class weights for loss function
        epsilon = 5120
        class_weights = [
            num_pixels / (background_class_freq + epsilon),
            num_pixels / (foreground_class_freq + epsilon),
            num_pixels / (road_class_freq + epsilon)
        ]

        # Getting ground truth data
        ground_truth = [
            vis,
            ground_truth_background,
            ground_truth_foreground,
            ground_truth_road
        ]

        return ground_truth, class_weights

    def getItemTrain(self, index):
        self.train_image = Image.open(str(self.train_images[index])).convert("RGB")
        self.train_label = Image.open(str(self.train_labels[index]))
        self.train_ground_truth, self.train_class_weights = \
            self.createGroundTruth(self.train_label)

        return np.array(self.train_image), self.train_ground_truth, \
            self.train_class_weights

    def getItemTrainPath(self, index):
        return str(self.train_images[index]), str(self.train_labels[index])

    def getItemVal(self, index):
        self.val_image = Image.open(str(self.val_images[index])).convert("RGB")
        self.val_label = Image.open(str(self.val_labels[index]))
        self.val_ground_truth, self.val_class_weights = \
            self.createGroundTruth(self.val_label)

        return np.array(self.val_image), self.val_ground_truth, \
            self.val_class_weights

    def getItemValPath(self, index):
        return str(self.val_images[index]), str(self.val_labels[index])
