#! /usr/bin/env python3

import cv2
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.object_seg_infer import ObjectSegNetworkInfer


# Class color map: class_id -> RGB
COLOR_MAP = {
    0: (61,  93, 255),   # background
    1: (255,  28, 145),  # man-made_structures
    2: (255,  61,  61),  # vulnerable_living
    3: (255, 190,  61),  # vehicle
    4: (  0, 100,   0),  # natural_obstacles
}

# Precompute BGR LUT for fast numpy colorization
_BGR_LUT = np.zeros((len(COLOR_MAP), 3), dtype=np.uint8)
for _cid, _rgb in COLOR_MAP.items():
    _BGR_LUT[_cid] = [_rgb[2], _rgb[1], _rgb[0]]


def make_visualization(prediction):
    """Convert (H,W) class-index array to (H,W,3) BGR image."""
    return _BGR_LUT[prediction]


def main():

    parser = ArgumentParser()
    parser.add_argument('-p', '--model_checkpoint_path', dest='model_checkpoint_path',
                        help='path to pytorch checkpoint file')
    parser.add_argument('-i', '--video_filepath', dest='video_filepath',
                        help='path to input video')
    parser.add_argument('-o', '--output_file', dest='output_file',
                        help='output video path (without extension)')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.5,
                        help='overlay transparency (default: 0.5)')
    parser.add_argument('-v', '--vis', action='store_true', default=False,
                        help='show live visualization while processing')
    args = parser.parse_args()

    model = ObjectSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print('ObjectSeg Model Loaded')

    cap = cv2.VideoCapture(args.video_filepath)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video_filepath}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = args.output_file + '.avi'
    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, (1280, 720))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Processing {frame_count} frames -> {output_path}')

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            print('Frame not read — ending processing')
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb).resize((640, 320))

        prediction = model.inference(image_pil)   # (H,W) class indices
        vis_bgr = make_visualization(prediction)  # (H,W,3) BGR

        frame   = cv2.resize(frame,   (1280, 720))
        vis_bgr = cv2.resize(vis_bgr, (1280, 720), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(vis_bgr, args.alpha, frame, 1 - args.alpha, 0)

        if args.vis:
            cv2.imshow('ObjectSeg Prediction', overlay)
            cv2.waitKey(10)

        writer.write(overlay)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print('Completed')


if __name__ == '__main__':
    main()
