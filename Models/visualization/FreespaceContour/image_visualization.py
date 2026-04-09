#! /usr/bin/env python3

import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../../')
from inference.freespace_contour_infer import FreespaceContourNetworkInfer


def make_visualization(image, contour_indices):
    height = image.shape[0]
    width = image.shape[1]
    
    # Reconstruct angles Left (Pi) to Right (0)
    num_rays = len(contour_indices)
    angles = np.linspace(np.pi, 0, num_rays)
    
    start_r = height - 1
    start_c = width // 2
    
    ray_slice_dist = 10 
    
    prev_pt = None
    for idx, angle in zip(contour_indices, angles):
        dist = idx * ray_slice_dist
        # Calculate coordinates
        # angle 0 (right): r same, c increases
        # angle pi/2 (up): r decreases, c same
        end_r = int(start_r - dist * np.sin(angle))
        end_c = int(start_c + dist * np.cos(angle))
        
        if 0 <= end_r < height and 0 <= end_c < width:
            curr_pt = (end_c, end_r)
            if prev_pt is not None:
                cv2.line(image, prev_pt, curr_pt, (0, 0, 255), thickness=2)
            cv2.circle(image, curr_pt, 2, (0, 0, 255), thickness=-1)
            prev_pt = curr_pt
        else:
            prev_pt = None

    return image

def main():

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path",
                        help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath",
                        help="path to input image which will be processed by ObjectSeg")
    args = parser.parse_args()

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    if not model_checkpoint_path:
        print("Please provide model checkpoint path using -p")
        return

    model = FreespaceContourNetworkInfer(checkpoint_path=model_checkpoint_path)
    print('FreespaceContour Model Loaded')

    # Transparency factor
    # alpha = 0.5 (Not blending segmentation mask here, drawing points directly on image)

    # Reading input image
    print('Reading Image')
    input_image_filepath = args.input_image_filepath
    if not input_image_filepath:
        print("Please provide input image path using -i")
        return

    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Could not load image: {input_image_filepath}")
        return

    # Resize to network input size
    image_resized = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Run inference and create visualization
    print('Running Inference and Creating Visualization')
    contour_indices = model.inference(image_pil)
    
    # Visualize on the resized image
    vis_obj = image_resized.copy()
    vis_obj = make_visualization(vis_obj, contour_indices)

    # Resize back to original size? Or display resized?
    # Original script resized visualization back to original frame size.
    # But points depend on 640x320 geometry. Scaling points is tricky without recalculating.
    # Drawing on 640x320 and then resizing back is easier.
    
    vis_obj_final = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]))
    
    # Maybe blend?
    # image_vis_obj = cv2.addWeighted(vis_obj_final, alpha, frame, 1 - alpha, 0)
    # But points are just green dots. Blending might make them faint.
    # Let's just show the result.
    
    cv2.imshow('Prediction Contour', vis_obj_final)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
