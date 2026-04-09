# Visualization

Scripts for running each network on a single image or a video and rendering
the predictions as overlays. Each network has its own subfolder with
`image_visualization.py` and `video_visualization.py`.

All scripts use the inference classes from `inference/` internally.

---

## FreespaceSeg — Drivable Area Segmentation

### Single Image

```bash
python FreespaceSeg/image_visualization.py \
  -p /path/to/checkpoint.pth \
  -i /path/to/image.jpg
```

Displays the input image blended (50% alpha) with the 2-class prediction mask:

| Class | Color |
|---|---|
| Non-drivable | Blue `(255, 93, 61)` |
| Drivable | Cyan `(220, 255, 0)` |

### Video

```bash
python FreespaceSeg/video_visualization.py \
  -p /path/to/checkpoint.pth \
  -i /path/to/input.mp4 \
  -o output_freespaceseg \
  [-v]
```

| Argument | Description |
|---|---|
| `-o` | Output base name — saves `{name}.avi` (MJPG codec) |
| `-v` | Show live preview window |

---

## FreespaceContour — Drivable Boundary Contour

### Single Image

```bash
python FreespaceContour/image_visualization.py \
  -p /path/to/checkpoint.pth \
  -i /path/to/image.jpg
```

Draws 37 red circles connected by lines tracing the predicted drivable
boundary. Each circle is placed at a distance (bin index × 10 px) along its
ray, sweeping left to right across the bottom of the image.

### Video

```bash
python FreespaceContour/video_visualization.py \
  -p /path/to/checkpoint.pth \
  -i /path/to/input.mp4 \
  -o output_contour.mp4
```

Output is saved as an MP4 at the original video resolution.

---

## Elevation — Terrain Elevation Map

Requires a camera parameters file (flattened 3×4 `[R|t]`, `.npy`, shape `(12,)`).

### Single Image

```bash
python Elevation/image_visualization.py \
  -p /path/to/checkpoint.pth \
  -i /path/to/image.jpg \
  -c /path/to/camera_params.npy \
  [--alpha 0.5]
```

Displays a side-by-side view:
- **Left**: input image blended with a jet-colormap elevation overlay
- **Right**: color bar with elevation scale from −0.5 m (bottom) to +1.5 m (top)

The elevation map is predicted at H/4 × W/4 (80×160 for a 320×640 input)
and upsampled back to the original image size for display.

### Video

```bash
python Elevation/video_visualization.py \
  -p /path/to/checkpoint.pth \
  -i /path/to/input.mp4 \
  -c /path/to/camera_params.npy \
  -o output_elevation.mp4
```

---

## AutoSpeed — Speed Limit / Object Detection

### Single Image

```bash
python AutoSpeed/image_visualization.py \
  -p /path/to/best.pt \
  -i /path/to/image.jpg
```

Draws colored bounding boxes for each detection:

| Class ID | Color |
|---|---|
| 1 | Red |
| 2 | Yellow |
| 3 | Cyan |

### Video

```bash
python AutoSpeed/video_visualization.py \
  -p /path/to/best.pt \        # or /path/to/model.onnx
  -i /path/to/input.mp4 \
  -o output_autospeed \
  [-v]
```

Automatically detects whether the checkpoint is a PyTorch (`.pt`) or ONNX
(`.onnx`) model and selects the appropriate inference backend.

| Argument | Description |
|---|---|
| `-o` | Output base name — saves `{name}.avi` |
| `-v` | Show live preview at 960 px width |
