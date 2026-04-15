# Training

Training scripts for all five networks. Each script handles multi-dataset
loading, gradient accumulation, TensorBoard logging, checkpointing, and
validation automatically.

---

## Dataset Structure

All scripts expect datasets arranged as:

```
{dataset_root}/
└── {DATASET_NAME}/
    └── {DATASET_NAME}/
        ├── images/           ← source images (JPEG/PNG)
        ├── gt_masks/         ← label masks (PNG, color-coded)
        └── camera_params/    ← (elevation only) .npy files, shape (12,)
```

Use `data_parsing/mask_editing.py` to convert third-party datasets into this
layout before training.

---

## train_object_seg.py — Object Segmentation

Trains `ObjectSegNetwork` for 5-class semantic segmentation.

**Classes:**

| ID | Name | Color |
|---|---|---|
| 0 | background | blue `(61, 93, 255)` |
| 1 | man-made_structures | pink `(255, 28, 145)` |
| 2 | vulnerable_living | red `(255, 61, 61)` |
| 3 | vehicle | light orange `(255, 190, 61)` |
| 4 | natural_obstacles | dark green `(0, 100, 0)` |

**Supported datasets:** `CaSSeD`, `Goose`, `OFFSED`, `ORFD`, `Rellis_3D`, `Yamaha_CMU`, `CARLA_Mining`

```bash
python train_object_seg.py \
  -r /path/to/dataset_root \
  -s /path/to/save_checkpoints/ \
  -m /path/to/pretrained_sceneseg.pth \
  -e 100 \
  --learning_rate 0.0001
```

Resume from checkpoint:

```bash
python train_object_seg.py \
  -r /path/to/dataset_root \
  -s /path/to/save_checkpoints/ \
  -c /path/to/objectseg.pth \
  -l \
  -e 100 -a 50
```

Train on a subset of datasets:

```bash
python train_object_seg.py \
  -r /path/to/dataset_root \
  -s /path/to/save_checkpoints/ \
  -d CARLA_Mining Goose \
  -e 100
```

| Argument | Default | Description |
|---|---|---|
| `-r, --root` | required | Dataset root directory |
| `-s, --model_save_root_path` | required | Checkpoint save directory |
| `-m, --pretrained_checkpoint_path` | — | Pre-trained SceneSeg backbone weights |
| `-c, --checkpoint_path` | — | Existing ObjectSeg checkpoint to resume from |
| `-l, --load_from_save` | false | Load weights from `-c` checkpoint |
| `-d, --datasets` | all | One or more dataset names to train on |
| `-e, --num_epochs` | 100 | Number of training epochs |
| `-a, --start_epoch` | 0 | Starting epoch (for resuming) |
| `--learning_rate` | 0.0001 | Learning rate |

**Validation metric**: mIoU (overall + per class, logged to TensorBoard)

---

## train_freespace_seg.py — Drivable Area Segmentation

Trains `FreespaceNetwork` for 2-class (drivable / non-drivable) segmentation.

```bash
python train_freespace_seg.py \
  -r /path/to/dataset_root \
  -s /path/to/save_checkpoints/ \
  -m /path/to/pretrained_sceneseg.pth \
  -e 100 \
  --learning_rate 0.0001
```

| Argument | Default | Description |
|---|---|---|
| `-r, --root` | required | Dataset root directory |
| `-s, --model_save_root_path` | required | Checkpoint save directory |
| `-m, --pretrained_checkpoint_path` | — | Pre-trained SceneSeg backbone weights |
| `-c, --checkpoint_path` | — | Existing FreespaceSeg checkpoint to resume from |
| `-l, --load_from_save` | false | Load weights from `-c` checkpoint |
| `-e, --num_epochs` | 100 | Number of training epochs |
| `-a, --start_epoch` | 0 | Starting epoch (for resuming) |
| `--learning_rate` | 0.0001 | Learning rate |

**Validation metric**: mIoU (overall, drivable, non-drivable)

---

## train_freespace_contour.py — Drivable Boundary Contour

Trains the contour prediction head alongside the segmentation head.

```bash
python train_freespace_contour.py \
  -r /path/to/dataset_root \
  -s /path/to/save_checkpoints/ \
  -m /path/to/pretrained_sceneseg.pth \
  -e 100
```

Arguments are identical to `train_freespace_seg.py`.

**Validation metrics**: contour accuracy, mean absolute error (MAE)

---

## train_elevation.py — Terrain Elevation Map

Trains `ElevationNetwork` to predict a per-pixel elevation distribution
from a single RGB image and camera extrinsics.

Elevation is discretized into **40 bins** spanning **−0.5 m to +1.5 m**
(0.05 m per bin). Output resolution is H/4 × W/4 (80×160 for a 320×640 input).

```bash
python train_elevation.py \
  -r /path/to/dataset_root \
  -s /path/to/save_checkpoints/ \
  -e 100 \
  --learning_rate 0.0001
```

| Argument | Default | Description |
|---|---|---|
| `-r, --root` | required | Dataset root directory |
| `-s, --model_save_root_path` | required | Checkpoint save directory |
| `-c, --checkpoint_path` | — | Checkpoint to resume from |
| `-e, --num_epochs` | 100 | Number of training epochs |
| `-a, --start_epoch` | 0 | Starting epoch (for resuming) |
| `--learning_rate` | 0.0001 | Learning rate |

The `camera_params/` folder must contain one `.npy` file per image with a
flattened 3×4 `[R|t]` matrix (12 values, `float32`).

**Validation metric**: Mean Absolute Error (MAE) in metres

---

## auto_speed_trainer.py — Speed Limit / Object Detection

Trains a YOLOv8-style detector. Supports single-GPU and multi-GPU
Distributed Data Parallel (DDP) training.

```bash
# Single GPU
python auto_speed_trainer.py \
  --dataset /path/to/detection_dataset \
  --batch-size 32 \
  --epochs 30 \
  --version n

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 auto_speed_trainer.py \
  --dataset /path/to/detection_dataset \
  --batch-size 32 \
  --epochs 30
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | Dataset root (must contain `images/train/` and `images/val/`) |
| `--batch-size` | 32 | Batch size per GPU |
| `--epochs` | 30 | Training epochs |
| `--input-size` | 640 | Input image size |
| `--version` | `n` | Model size: `n` (nano), `s`, `m`, `l`, `x` |
| `--runs_dir` | `runs/` | Directory for checkpoints and TensorBoard logs |

Checkpoints are saved to `runs/run{N}/weights/best.pt` and `last.pt`.

**Validation metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall

---

## Training Loop Behaviour (all segmentation / elevation scripts)

**Batch size schedule** (automatically applied by epoch):

| Epoch range | Batch size |
|---|---|
| 0 – 9 | 16 |
| 10 – 29 | 8 |
| 30 – 59 | 4 |
| 60 – 79 | 2 |
| 80+ | 1 |

**Logging intervals** (steps):

| Every N steps | Action |
|---|---|
| 250 | Log training loss to TensorBoard |
| 1000 | Save visualization image to `training/` |
| 8000 | Save checkpoint + run full validation |

TensorBoard logs are written to `runs/`. To view:

```bash
tensorboard --logdir runs/
```
