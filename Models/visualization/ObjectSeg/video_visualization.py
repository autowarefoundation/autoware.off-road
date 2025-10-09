#! /usr/bin/env python3

import cv2
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.object_seg_infer import ObjectSegNetworkInfer
from data_utils.boundary_dists import BoundaryDists


def make_visualization(prediction, show_road=False):

    # Creating visualization object
    shape = prediction.shape
    row = shape[0]
    col = shape[1]
    vis_predict_object = np.zeros((row, col, 3), dtype="uint8")

    # Assigning background colour
    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61

    # Getting foreground object labels
    foreground_lables = np.where(prediction == 1)

    # Assigning foreground objects colour
    vis_predict_object[foreground_lables[0], foreground_lables[1], 0] = 145
    vis_predict_object[foreground_lables[0], foreground_lables[1], 1] = 28
    vis_predict_object[foreground_lables[0], foreground_lables[1], 2] = 255

    # Getting drivable_lables labels
    drivable_lables = np.where(prediction == 2)

    # Assigning drivable_lables colour
    vis_predict_object[drivable_lables[0], drivable_lables[1], 0] = 220
    vis_predict_object[drivable_lables[0], drivable_lables[1], 1] = 255
    vis_predict_object[drivable_lables[0], drivable_lables[1], 2] = 0

    return vis_predict_object

def boundry_dist_visualization(image, boundary_dists):
    height = image.shape[0]
    width = image.shape[1]
    cols = np.linspace(0, width - 1, len(boundary_dists), dtype=int)
    for col, dist in zip(cols, boundary_dists):
        row = int(height - 1 - dist * height)
        cv2.circle(image, (col, row), 5, (0, 100, 0), thickness=-1)

def main():

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path",
                        help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--video_filepath", dest="video_filepath",
                        help="path to input video which will be processed by ObjectSeg")
    parser.add_argument("-o", "--output_file", dest="output_file",
                        help="path to output video visualization file, must include output file name")
    parser.add_argument('-v', "--vis", action='store_true', default=False,
                        help="flag for whether to show frame by frame visualization while processing is occuring")
    parser.add_argument('-r', "--show_road", action='store_true', default=False,
                        help="flag for whether to show road segmentation in visualization")
    args = parser.parse_args()

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = ObjectSegNetworkInfer(checkpoint_path=model_checkpoint_path)
    print('ObjectSeg Model Loaded')

    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    video_filepath = args.video_filepath
    cap = cv2.VideoCapture(video_filepath)

    # Output filepath
    output_filepath_obj = args.output_file + '.avi'
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Video writer object
    writer_obj = cv2.VideoWriter(output_filepath_obj,
                                 cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))

    # Check if video catpure opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
    else:
        print('Reading video frames')

    # Transparency factor
    alpha = 0.5

    print('Processing started')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print('Frame not read - ending processing')
            break

        # Display the resulting frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize((640, 320))

        # Running inference
        prediction = model.inference(image_pil)
        vis_obj = make_visualization(prediction, args.show_road)

        # Boundary distance visualization
        boundry = BoundaryDists(prediction, 32)
        boundry_dists = boundry.get_boundary_dists()
        boundry_dist_visualization(vis_obj, boundry_dists)

        # Resizing to match the size of the output video
        # which is set to standard HD resolution
        frame = cv2.resize(frame, (1280, 720))
        vis_obj = cv2.resize(vis_obj, (1280, 720))

        # Create the composite visualization
        image_vis_obj = cv2.addWeighted(
            vis_obj, alpha, frame, 1 - alpha, 0)

        if args.vis:
            cv2.imshow('Prediction Objects', image_vis_obj)
            cv2.waitKey(10)

        # Writing to video frame
        writer_obj.write(image_vis_obj)

    # When everything done, release the video capture and writer objects
    cap.release()
    writer_obj.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print('Completed')


if __name__ == '__main__':
    main()
