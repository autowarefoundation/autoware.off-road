import torch
from torchvision import transforms, ops
from PIL import Image
import numpy as np
import onnxruntime as ort
import os

class AutoSpeedNetworkInfer():
    def __init__(self, checkpoint_path=''):
        self.train_size = (640, 640)  # target width, height
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f'Using {self.device} for inference')

        # Load model
        if os.path.isfile(checkpoint_path):
            actual_path = checkpoint_path
        else:
            actual_path = os.path.join(checkpoint_path, "best.pt")
            
        self.model = torch.load(actual_path, map_location="cpu", weights_only=False)['model']
        self.model = self.model.to(self.device).eval()

    def resize_letterbox(self, img: Image.Image):
        """
        Resize image maintaining aspect ratio with padding.
        Returns:
            padded_img: PIL.Image, resized + padded
            scale: float, scaling factor applied to original image
            pad_x: int, horizontal padding
            pad_y: int, vertical padding
        """
        target_w, target_h = self.train_size
        orig_w, orig_h = img.size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        padded_img = Image.new("RGB", self.train_size, (114, 114, 114))  # gray padding

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded_img.paste(img_resized, (pad_x, pad_y))

        return padded_img, scale, pad_x, pad_y

    def image_to_tensor(self, image: Image.Image):
        """Convert PIL image to tensor and keep scale/padding info."""
        img, scale, pad_x, pad_y = self.resize_letterbox(image)
        tensor = transforms.ToTensor()(img).to(self.device).half()
        return tensor.unsqueeze(0), scale, pad_x, pad_y

    def xywh2xyxy(self, x):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    def nms(self, preds, iou_thres=0.45):
        """Apply NMS on predictions tensor [x1,y1,x2,y2,score,class]"""
        if preds.numel() == 0:
            return torch.empty(0, 6)
        boxes = preds[:, :4]
        scores = preds[:, 4]
        keep = ops.nms(boxes, scores, iou_thres)
        return preds[keep]

    def post_process_predictions(self, raw_predictions, conf_thres=0.6, iou_thres=0.45):
        predictions = raw_predictions.squeeze(0).permute(1, 0)
        boxes = predictions[:, :4]  # model gives cx,cy,w,h
        class_probs = predictions[:, 4:]

        # --- confidence filter ---
        scores, class_ids = torch.max(class_probs.sigmoid(), dim=1)
        mask = scores > conf_thres
        if mask.sum() == 0:
            return torch.empty(0, 6)

        # --- convert to xyxy before NMS ---
        boxes_xyxy = self.xywh2xyxy(boxes[mask])

        combined = torch.cat([
            boxes_xyxy,
            scores[mask].unsqueeze(1),
            class_ids[mask].float().unsqueeze(1)
        ], dim=1)

        return self.nms(combined, iou_thres)

    def inference(self, image: Image.Image):
        """
        Run inference on a single PIL image.
        Returns list of predictions: [[x1,y1,x2,y2,score,class], ...] in original image coordinates
        """
        orig_w, orig_h = image.size
        image_tensor, scale, pad_x, pad_y = self.image_to_tensor(image)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        predictions = self.post_process_predictions(predictions)
        if predictions.numel() == 0:
            return []

        # --- adjust from letterboxed to original coords ---
        predictions[:, [0, 2]] = (predictions[:, [0, 2]] - pad_x) / scale
        predictions[:, [1, 3]] = (predictions[:, [1, 3]] - pad_y) / scale

        # clamp to image bounds
        predictions[:, [0, 2]] = predictions[:, [0, 2]].clamp(0, orig_w)
        predictions[:, [1, 3]] = predictions[:, [1, 3]].clamp(0, orig_h)

        return predictions.tolist()

class AutoSpeedONNXInfer:
    def __init__(self, onnx_path):
        self.train_size = (640, 640)
        available_providers = ort.get_available_providers()
        requested_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in requested_providers if p in available_providers]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    def resize_letterbox(self, img: Image.Image):
        target_w, target_h = self.train_size
        orig_w, orig_h = img.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        padded_img = Image.new("RGB", self.train_size, (114, 114, 114))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded_img.paste(img_resized, (pad_x, pad_y))
        return padded_img, scale, pad_x, pad_y

    def image_to_array(self, image: Image.Image):
        img, scale, pad_x, pad_y = self.resize_letterbox(image)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]
        return img_array, scale, pad_x, pad_y

    def xywh2xyxy(self, boxes):
        x = boxes.copy()
        x[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        x[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        x[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        x[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return x

    def nms(self, boxes, scores, iou_threshold=0.45):
        boxes_t = torch.from_numpy(boxes)
        scores_t = torch.from_numpy(scores)
        keep = ops.nms(boxes_t, scores_t, iou_threshold)
        return keep.numpy()

    def post_process_predictions(self, raw_predictions, conf_thres=0.6, iou_thres=0.45):
        predictions = raw_predictions.squeeze(0).T
        boxes = predictions[:, :4]
        class_probs = predictions[:, 4:]
        
        # Sigmoid + confidence filter
        class_probs = 1 / (1 + np.exp(-class_probs))
        scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        mask = scores > conf_thres
        
        if mask.sum() == 0:
            return []
        
        boxes_xyxy = self.xywh2xyxy(boxes[mask])
        scores_filtered = scores[mask]
        class_ids_filtered = class_ids[mask]
        
        # NMS
        keep = self.nms(boxes_xyxy, scores_filtered, iou_thres)
        
        results = []
        for idx in keep:
            results.append([
                float(boxes_xyxy[idx, 0]), float(boxes_xyxy[idx, 1]),
                float(boxes_xyxy[idx, 2]), float(boxes_xyxy[idx, 3]),
                float(scores_filtered[idx]), float(class_ids_filtered[idx])
            ])
        return results

    def inference(self, image: Image.Image):
        orig_w, orig_h = image.size
        img_array, scale, pad_x, pad_y = self.image_to_array(image)
        
        outputs = self.session.run(None, {'input': img_array})
        predictions = self.post_process_predictions(outputs[0])
        
        if len(predictions) == 0:
            return []
        
        # Adjust coordinates
        for pred in predictions:
            pred[0] = (pred[0] - pad_x) / scale
            pred[1] = (pred[1] - pad_y) / scale
            pred[2] = (pred[2] - pad_x) / scale
            pred[3] = (pred[3] - pad_y) / scale
            # Clamp to bounds
            pred[0] = max(0, min(orig_w, pred[0]))
            pred[1] = max(0, min(orig_h, pred[1]))
            pred[2] = max(0, min(orig_w, pred[2]))
            pred[3] = max(0, min(orig_h, pred[3]))
        
        return predictions
