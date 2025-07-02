import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


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