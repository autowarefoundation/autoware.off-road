#! /usr/bin/env python3

import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.freespace_seg_infer import FreespaceSegNetworkInfer
from data_utils.radial_contour_dists import RadialContourDists


def make_visualization(prediction):

    # Creating visualization object
    shape = prediction.shape
    row = shape[0]
    col = shape[1]
    vis_predict_object = np.zeros((row, col, 3), dtype="uint8")

    # Assigning non_drivable colour
    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61

    drivable_lables = np.where(prediction == 1)

    # Assigning drivable_color
    vis_predict_object[drivable_lables[0], drivable_lables[1], 0] = 220
    vis_predict_object[drivable_lables[0], drivable_lables[1], 1] = 255
    vis_predict_object[drivable_lables[0], drivable_lables[1], 2] = 0

    return vis_predict_object

def main():

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path",
                        help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath",
                        help="path to input image which will be processed by ObjectSeg")
    args = parser.parse_args()

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = FreespaceSegNetworkInfer(checkpoint_path=model_checkpoint_path)
    print('FreespaceSeg Model Loaded')

    # Transparency factor
    alpha = 0.5

    # Reading input image
    print('Reading Image')
    input_image_filepath = args.input_image_filepath
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image = cv2.resize(image, (640, 320), interpolation=cv2.INTER_LINEAR)
    image_pil = Image.fromarray(image)

    # Run inference and create visualization
    print('Running Inference and Creating Visualization')
    prediction = model.inference(image_pil)
    vis_obj = make_visualization(prediction)

    # Resize and display visualization
    vis_obj = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]))
    image_vis_obj = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)
    cv2.imshow('Prediction Objects', image_vis_obj)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
