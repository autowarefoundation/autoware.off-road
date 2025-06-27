import torch
from tqdm import tqdm

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
