import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
import pandas as pd


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def read_label_mapping(csv_path):
    """
    Read CSV file and create label mappings.
    
    Args:
        csv_path (str): Path to the CSV file containing label mappings
        
    Returns:
        tuple: (label_id_to_name, rgb_to_label_id)
            - label_id_to_name: dict with label_id as key and label_name as value
            - rgb_to_label_id: dict with RGB tuple as key and label_id as value
    """
    df = pd.read_csv(csv_path)
    label_id_to_name = {}
    for _, row in df.iterrows():
        label_id = row['label_key']
        label_name = row['class_name']
        label_id_to_name[label_id] = label_name
    
    rgb_to_label_id = {}
    for _, row in df.iterrows():
        hex_color = row['hex']
        label_id = row['label_key']
        rgb_tuple = hex_to_rgb(hex_color)
        rgb_to_label_id[rgb_tuple] = label_id
    
    return label_id_to_name, rgb_to_label_id


def get_mean_std(dataset):
    psum     = torch.zeros(3) 
    psum_sq  = torch.zeros(3)
    n_pixels = 0
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        img_t = sample["image"]
        psum += img_t.sum(dim=(1, 2))
        psum_sq += (img_t ** 2).sum(dim=(1, 2))
        n_pixels += img_t.shape[1] * img_t.shape[2]

    mean = psum / n_pixels
    std = torch.sqrt((psum_sq / n_pixels) - mean ** 2)

    return mean, std


def build_segformer_train_augs(
    crop_size=(512,512),
    scale_base=(2048,512),
    ratio_range=(0.5,2.0),
    mean=(123.675,116.28,103.53),
    std=(58.395,57.12,57.375)
):

    long, short = scale_base
    augments = A.Compose([
        A.LongestMaxSize(max_size=long, interpolation=1),
        A.SmallestMaxSize(max_size=short, interpolation=1),
        A.RandomScale(scale_limit=(ratio_range[0]-1.0, ratio_range[1]-1.0), p=1.0),
        A.RandomCrop(width=crop_size[0], height=crop_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})

    return augments