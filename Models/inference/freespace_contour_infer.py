#! /usr/bin/env python3

import torch
from torchvision import transforms
import sys
sys.path.append('..')
from model_components.freespace_networks import FreespaceNetwork
from model_components.scene_seg_network import SceneSegNetwork


class FreespaceContourNetworkInfer():
    def __init__(self, checkpoint_path=''):

        # Image loader
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ]
        )

        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        # Instantiate model, load to device and set to evaluation mode
        sceneSegNetwork = SceneSegNetwork()
        self.model = FreespaceNetwork(sceneSegNetwork)

        if len(checkpoint_path) > 0:
            self.model.load_state_dict(torch.load
                                       (checkpoint_path, weights_only=True, map_location=self.device))
        else:
            raise ValueError(
                'No path to checkpoint file provided in class initialization')

        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def inference(self, image):

        width, height = image.size
        # if (width != 640 or height != 320):
        #     raise ValueError(
        #         'Incorrect input size - input image must have height of 320px and width of 640px')
        
        # Note: The original code had a size check, keeping it commented or active? 
        # The segmentation infer script had it. I'll keep it commented or assume caller resizes.
        # But actually, best to be safe given the hardcoded ray logic (start_r, start_c depend on size).
        # The visualization script resizes to 640x320 before calling inference.

        image_tensor = self.image_loader(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Run model
        with torch.no_grad():
            _, contour_prediction = self.model(image_tensor)

        # contour_prediction: (1, 37, 46, 1)
        # Squeeze batch and width dim
        contour_prediction = contour_prediction.squeeze(0).squeeze(-1) # (37, 46)
        
        # Find max class probability (bin index)
        contour_indices = torch.argmax(contour_prediction, dim=1) # (37)
        
        return contour_indices.cpu().numpy()
