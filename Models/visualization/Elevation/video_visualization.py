#! /usr/bin/env python3

import cv2
import sys
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from model_components.elevation_network import ElevationNetwork
from visualization.Elevation.image_visualization import make_visualization, run_inference
from torchvision import transforms
import torch


_H_MIN = -0.50
_H_MAX =  1.50


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--model_checkpoint_path', dest='model_checkpoint_path', required=True,
                        help='path to ElevationNetwork .pth checkpoint')
    parser.add_argument('-i', '--input_video_filepath', dest='input_video_filepath', required=True,
                        help='path to input video file (or 0 for webcam)')
    parser.add_argument('-c', '--camera_params_filepath', dest='camera_params_filepath', required=True,
                        help='path to .npy file containing the (12,) [R|t] camera parameter vector')
    parser.add_argument('-o', '--output_video_filepath', dest='output_video_filepath', default='',
                        help='optional path to save the output video (.mp4)')
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

    # Camera params (constant for the whole video)
    camera_params = np.load(args.camera_params_filepath).astype(np.float32)

    # Open video source
    src = 0 if args.input_video_filepath == '0' else args.input_video_filepath
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {src}')

    # Optional output writer
    writer = None
    if args.output_video_filepath:
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output_video_filepath, fourcc, fps,
                                 (w + 30, h))   # +30 for colour bar

    # Build colour bar once
    bar_h, bar_w = None, 30

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if bar_h is None:
            bar_h = frame.shape[0]
            bar_norm = np.linspace(1.0, 0.0, bar_h)[:, None]
            bar_rgb  = (cm.jet(bar_norm)[:, :, :3] * 255).astype(np.uint8)
            bar_bgr  = cv2.cvtColor(np.repeat(bar_rgb, bar_w, axis=1), cv2.COLOR_RGB2BGR)
            cv2.putText(bar_bgr, f'{_H_MAX:.1f}m', (2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(bar_bgr, f'{_H_MIN:.1f}m', (2, bar_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Pre-process
        image_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (640, 320), interpolation=cv2.INTER_LINEAR)
        image_pil     = Image.fromarray(image_resized)

        # Inference
        elev_map = run_inference(model, image_pil, camera_params, device)

        # Visualize
        vis = make_visualization(elev_map)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        vis = cv2.resize(vis, (frame.shape[1], frame.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

        overlay  = cv2.addWeighted(vis, args.alpha, frame, 1.0 - args.alpha, 0)
        combined = np.hstack([overlay, bar_bgr])

        cv2.imshow('Elevation Map', combined)
        if writer is not None:
            writer.write(combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
