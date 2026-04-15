#! /usr/bin/env python3

import cv2
import sys
import numpy as np
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
    parser.add_argument('-i', '--input_image_filepath', dest='input_image_filepath',
                        help='path to input image')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.5,
                        help='overlay transparency (default: 0.5)')
    args = parser.parse_args()

    model = ObjectSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print('ObjectSeg Model Loaded')

    frame = cv2.imread(args.input_image_filepath, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f'Could not read image: {args.input_image_filepath}')

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize((640, 320))

    print('Running Inference')
    prediction = model.inference(image_pil)   # (H,W) class indices at 640x320
    vis_bgr = make_visualization(prediction)  # (H,W,3) BGR

    # Scale visualization to original frame size
    vis_bgr = cv2.resize(vis_bgr, (frame.shape[1], frame.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(vis_bgr, args.alpha, frame, 1 - args.alpha, 0)

    cv2.imshow('ObjectSeg Prediction', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
