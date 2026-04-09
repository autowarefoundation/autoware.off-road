#! /usr/bin/env python3

import cv2
import sys
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from model_components.elevation_network import ElevationNetwork
from torchvision import transforms
import torch
import numpy as np


# Elevation bin constants — must match elevation_head.py
_H_MIN  = -0.50   # metres
_H_MAX  =  1.50   # metres


def make_visualization(elevation_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Convert a (H, W) float elevation map (metres) to a jet-coloured RGB uint8 image.

    Args:
        elevation_map: expected elevation in metres, shape (H, W).
        alpha:         unused here; kept for API consistency with overlay callers.

    Returns:
        RGB uint8 array (H, W, 3).
    """
    norm = np.clip((elevation_map - _H_MIN) / (_H_MAX - _H_MIN), 0.0, 1.0)
    rgb  = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def run_inference(model, image_pil: Image.Image, camera_params: np.ndarray,
                  device: torch.device) -> np.ndarray:
    """Run ElevationNetwork and return the expected elevation map in metres.

    Args:
        model:         ElevationNetwork (eval mode, on device).
        image_pil:     RGB PIL image, already resized to (640 × 320).
        camera_params: float32 (12,) flattened [R|t] vector.
        device:        torch device.

    Returns:
        elevation_map: (H/8, W/8) float32 array in metres.
    """
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_t = loader(image_pil).unsqueeze(0).to(device)
    cam_t = torch.from_numpy(camera_params).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model(img_t, cam_t)                              # (1, 40, H/8, W/8)
        elev  = model.ElevationHead.expected_elevation(probs)    # (1, H/8, W/8)

    return elev.squeeze(0).cpu().numpy()                         # (H/8, W/8)


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--model_checkpoint_path', dest='model_checkpoint_path', required=True,
                        help='path to ElevationNetwork .pth checkpoint')
    parser.add_argument('-i', '--input_image_filepath', dest='input_image_filepath', required=True,
                        help='path to the input RGB image')
    parser.add_argument('-c', '--camera_params_filepath', dest='camera_params_filepath', required=True,
                        help='path to .npy file containing the (12,) [R|t] camera parameter vector')
    parser.add_argument('--alpha', dest='alpha', default=0.5, type=float,
                        help='overlay blending factor (0 = image only, 1 = elevation only)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = ElevationNetwork()
    model.load_state_dict(torch.load(args.model_checkpoint_path, weights_only=True,
                                     map_location=device))
    model.to(device).eval()
    print('ElevationNetwork loaded')

    # Load inputs
    frame          = cv2.imread(args.input_image_filepath, cv2.IMREAD_COLOR)
    image_rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized  = cv2.resize(image_rgb, (640, 320), interpolation=cv2.INTER_LINEAR)
    image_pil      = Image.fromarray(image_resized)
    camera_params  = np.load(args.camera_params_filepath).astype(np.float32)

    # Inference
    elev_map = run_inference(model, image_pil, camera_params, device)   # (40, 80)

    # Colourmap → resize back to frame resolution for overlay
    vis = make_visualization(elev_map)                                   # (40, 80, 3)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    vis = cv2.resize(vis, (frame.shape[1], frame.shape[0]),
                     interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(vis, args.alpha, frame, 1.0 - args.alpha, 0)

    # Colour bar legend (jet, −0.5 → +1.5 m)
    bar_h, bar_w = frame.shape[0], 30
    bar_norm  = np.linspace(1.0, 0.0, bar_h)[:, None]
    bar_rgb   = (cm.jet(bar_norm)[:, :, :3] * 255).astype(np.uint8)
    bar_bgr   = cv2.cvtColor(np.repeat(bar_rgb, bar_w, axis=1), cv2.COLOR_RGB2BGR)
    cv2.putText(bar_bgr, f'{_H_MAX:.1f}m', (2, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(bar_bgr, f'{_H_MIN:.1f}m', (2, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    combined = np.hstack([overlay, bar_bgr])
    cv2.imshow('Elevation Map', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
