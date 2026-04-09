# Inference

Python classes for running each trained network on a single image. Import
these into any application or ROS node — they handle model loading, resizing,
normalization, and post-processing internally.

All segmentation models expect input as a **PIL Image** and resize it to
640×320 internally. The AutoSpeed detector letterboxes input to 640×640.

---

## FreespaceSegNetworkInfer — Drivable Area Segmentation

`freespace_seg_infer.py`

Predicts a 2-class mask: `0 = non-drivable`, `1 = drivable`.

```python
from inference.freespace_seg_infer import FreespaceSegNetworkInfer
from PIL import Image

model = FreespaceSegNetworkInfer(checkpoint_path='/path/to/checkpoint.pth')
image = Image.open('frame.jpg')
mask = model.inference(image)   # numpy (H, W), values 0 or 1
```

---

## FreespaceContourNetworkInfer — Drivable Boundary Contour

`freespace_contour_infer.py`

Predicts 37 bin indices describing the drivable boundary as a radial contour,
sampled from left (π rad) to right (0 rad) relative to the vehicle forward
direction.

```python
from inference.freespace_contour_infer import FreespaceContourNetworkInfer
from PIL import Image

model = FreespaceContourNetworkInfer(checkpoint_path='/path/to/checkpoint.pth')
image = Image.open('frame.jpg')
contour = model.inference(image)  # numpy (37,), bin indices
```

Each bin index × `ray_slice_dist` (10 px) gives the distance along that ray.
Reconstruct 2-D points with:

```python
import numpy as np

h, w = 320, 640
start_r, start_c = h - 1, w // 2
angles = np.linspace(np.pi, 0, 37)
dists  = contour * 10  # pixels

xs = start_c + dists * np.cos(angles)
ys = start_r - dists * np.sin(angles)
```

---

## ObjectSegNetworkInfer — Multi-class Object Segmentation

`object_seg_infer.py`

Predicts a per-pixel class index mask for off-road object categories.

```python
from inference.object_seg_infer import ObjectSegNetworkInfer
from PIL import Image

model = ObjectSegNetworkInfer(checkpoint_path='/path/to/checkpoint.pth')
image = Image.open('frame.jpg')
mask = model.inference(image)   # numpy (H, W), class indices
```

---

## AutoSpeedNetworkInfer — Speed Limit / Object Detection

`auto_speed_infer.py`

YOLOv8-style detector. Accepts a `.pt` checkpoint path or a directory
containing `best.pt`. Also supports ONNX via `AutoSpeedONNXInfer`.

```python
from inference.auto_speed_infer import AutoSpeedNetworkInfer
from PIL import Image

model = AutoSpeedNetworkInfer(checkpoint_path='/path/to/best.pt')
image = Image.open('frame.jpg')
detections = model.inference(image)
# detections: list of [x1, y1, x2, y2, score, class_id]
```

For ONNX runtime inference:

```python
from inference.auto_speed_infer import AutoSpeedONNXInfer

model = AutoSpeedONNXInfer(checkpoint_path='/path/to/model.onnx')
detections = model.inference(image)
```

Default thresholds: confidence `0.6`, NMS IoU `0.45`.

---

## Notes

- All models auto-select CUDA if available, otherwise fall back to CPU.
- Normalization uses ImageNet statistics:
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- Checkpoint format is `torch.load(..., weights_only=True)`.
