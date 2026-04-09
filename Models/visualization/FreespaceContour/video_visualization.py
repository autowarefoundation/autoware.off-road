#! /usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import sys
import time
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
    
    # Draw on a copy or directly? Copy is safer.
    vis_image = image.copy()
    
    prev_pt = None
    for idx, angle in zip(contour_indices, angles):
        dist = idx * ray_slice_dist
        # Calculate coordinates
        end_r = int(start_r - dist * np.sin(angle))
        end_c = int(start_c + dist * np.cos(angle))
        
        if 0 <= end_r < height and 0 <= end_c < width:
            curr_pt = (end_c, end_r)
            if prev_pt is not None:
                cv2.line(vis_image, prev_pt, curr_pt, (0, 0, 255), thickness=2)
            cv2.circle(vis_image, curr_pt, 2, (0, 0, 255), thickness=-1)
            prev_pt = curr_pt
        else:
            prev_pt = None

    return vis_image

def main():

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path",
                        help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_video_filepath", dest="input_video_filepath",
                        help="path to input video which will be processed by ObjectSeg")
    parser.add_argument("-o", "--output_video_filepath", dest="output_video_filepath",
                        help="path to output video which will be processed by ObjectSeg")
    args = parser.parse_args()

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    if not model_checkpoint_path:
        print("Please provide model checkpoint path using -p")
        return

    model = FreespaceContourNetworkInfer(checkpoint_path=model_checkpoint_path)
    print('FreespaceContour Model Loaded')

    # Reading input video
    print('Reading Video')
    input_video_filepath = args.input_video_filepath
    if not input_video_filepath:
        print("Please provide input video path using -i")
        return
        
    cap = cv2.VideoCapture(input_video_filepath)
    if not cap.isOpened():
        print(f"Error opening video file {input_video_filepath}")
        return

    # Output video
    output_video_filepath = args.output_video_filepath
    if not output_video_filepath:
         # Default output name if not provided
         output_video_filepath = "output_contour.mp4"

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Force 30 fps? Or keep original.
    
    out = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    print('Processing Frames...')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize options
            image_resized = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_LINEAR)
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Inference
            contour_indices = model.inference(image_pil)
            
            # Visualization on resized frame
            vis_frame = make_visualization(image_resized, contour_indices)
            
            # Resize back to original
            final_frame = cv2.resize(vis_frame, (frame_width, frame_height))
            
            # Write the frame
            out.write(final_frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_video_filepath}")

if __name__ == '__main__':
    main()
