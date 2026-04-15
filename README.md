# Autoware — Off-road Autonomy Pilot

The goal of this open-source Autoware project is to extend Autoware stacks and pipelines to enable safe and reliable off-road autonomous driving for both terrestrial and Mars/Moon vehicles, with a focus on achieving fully integrated end-to-end autonomy. New off-road perception, planning/control, and simulation stacks are developed to support autonomous navigation over rugged terrain on a wide variety of surfaces.

---

## Use Cases

| Category | Applications |
|---|---|
| **Terrestrial** | Off-road mining trucks, racing buggies, agricultural vehicles |
| **Mars / Moon** | Mars rovers, Lunar Terrain Vehicles (LTVs) in GNSS-denied environments |

**Terrestrial** vehicles can carry a full sensor suite (LiDAR, cameras, GNSS).
**Planetary** vehicles rely primarily on vision-based navigation and localisation due to harsh, GNSS-denied environments.

The Off-road Autonomy Pilot is currently focused on the **terrestrial** use cases, with focuses on **mining truck** and **RoboRacer off-road racing**.

---

## Repository Structure

```
autoware.off-road/
├── Models/          ← Perception networks (training, inference, visualization)
├── Modules/         ← ROS 2 planning and control nodes
├── Simulation/      ← CARLA simulation environments and scripts
├── Demos/
├── Diagrams/
└── Media/
```

---

## Models

Vision-based perception networks for off-road driving. All segmentation models
use a shared **EfficientNet-B0** backbone with a **SceneContext** attention
module and **SceneNeck** FPN decoder producing stride-4 features (256 ch).

| Network | Task | Input | Output |
|---|---|---|---|
| **FreespaceSegNetwork** | Drivable area segmentation | 640×320 RGB | 2-class mask (drivable / non-drivable) |
| **FreespaceContourNetwork** | Drivable boundary contour | 640×320 RGB | 37 radial bin indices |
| **ElevationNetwork** | Terrain elevation map | 640×320 RGB + 12-value camera extrinsics | (B, 40, 80, 160) elevation bin probabilities |
| **AutoSpeedNetwork** | Speed sign / object detection | 640×640 RGB | Bounding boxes with class and confidence |

### Models Subdirectories

| Folder | Description |
|---|---|
| `model_components/` | Individual network building blocks (backbone, necks, heads) |
| `data_parsing/` | Interactive SAM annotation tool and batch dataset preparation — see [data_parsing/README.md](Models/data_parsing/README.md) |
| `training/` | Training scripts for all four networks — see [training/README.md](Models/training/README.md) |
| `inference/` | Inference classes for deployment and ROS integration — see [inference/README.md](Models/inference/README.md) |
| `visualization/` | Image and video visualization scripts — see [visualization/README.md](Models/visualization/README.md) |

### Data Parsing

Two tools prepare training data for all networks:

**`mask_labeling.py`** — Interactive SAM-assisted annotation tool. Short-click to auto-segment a region with SAM, hold-click to paint manually, middle-click to erase, Ctrl+click to flood-fill. Press `0`–`9` to switch class, `Z` to undo, `N`/`B` to navigate images, `ESC` to save. Outputs one color-coded PNG mask per image.

**`mask_editing.py`** — Batch dataset preparation using Hydra configuration. Converts third-party off-road datasets (Rellis-3D, GOOSE, CASSED, OFFSED, Yamaha-CMU, ORFD) into the standard layout expected by all training scripts. Remaps source label colors/IDs to project class IDs and resizes images.

Output structure for all datasets:

```
{dataset_root}/{DATASET}/{DATASET}/
├── images/          ← resized source images (JPEG/PNG)
├── gt_masks/        ← remapped color masks (PNG)
└── camera_params/   ← (elevation only) flattened [R|t] arrays (.npy, shape (12,))
```

See [data_parsing/README.md](Models/data_parsing/README.md) for installation instructions (includes SAM setup) and full usage.

### ElevationNetwork

Predicts a per-pixel terrain height distribution from a single RGB image and
camera extrinsics. Output is 40 softmax bins spanning −0.5 m to +1.5 m at
5 cm resolution, at H/4 × W/4 (80×160 for a 320×640 input). Call
`ElevationHead.expected_elevation(probs)` to collapse to a scalar height map
in metres.

---

## Simulation

CARLA-based simulation environment using a mining truck (`vehicle.miningtruck.miningtruck`) on the `Mine_01` map.

### Scripts

| Script | Description |
|---|---|
| `Simulation/CARLA/scripts/mining_sim.py` | Manual control sim with keyboard input, ROS 2 Ackermann bridge, and elevation dataset recording |

### Key Features

- **Manual control** — WASD / arrow keys, autopilot toggle (`P`), Ackermann control (`F`)
- **ROS 2 bridge** — subscribes to `/carla/ego_vehicle/ackermann_cmd`, publishes `/carla/ego_vehicle/speed`
- **Elevation dataset recording** — press `E` to toggle; saves paired RGB + LiDAR ground-truth elevation maps to `Simulation/CARLA/data/_elevation_dataset/`
- **Semantic segmentation dataset recording** — press `J` to toggle; saves paired RGB images and CARLA semantic segmentation ground-truth masks to the destination directory defined in `carla_objectseg.yaml`

### Elevation Dataset Recording

Dedicated RGB camera and LiDAR are mounted at the same position as the default
display camera. On each LiDAR tick, if a recent RGB frame is available, a pair
is saved:

```
Simulation/CARLA/data/_elevation_dataset/
├── rgb/           ← 640×320 PNG
└── elevation/     ← 640×320 float32 .npy  (z-height in metres, NaN = no data)
```

---

## Demos

### CARLA Mining Demo

Closed-loop autonomy stack running inside CARLA on the `Mine_01` map. Subscribes to the ego vehicle's camera, runs perception inference, and publishes `AckermannDrive` commands at ~10 Hz.

#### Prerequisites

1. Install CARLA-0.10.0
2. Build the C++ modules:
    ```bash
    cd Modules/GapFollower && mkdir -p build && cd build && cmake .. && make
    cd Modules/Control/Steering/SteeringController && mkdir -p build && cd build && cmake .. && make
    cd Modules/Control/Longitudinal/PIController && mkdir -p build && cd build && cmake .. && make
    ```

#### Launch

**Terminal 1 — CARLA Core:**
```bash
./Carla-0.10.0-Linux-Shipping/CarlaUnreal.sh --ros2
```

**Terminal 2 — CARLA Mining Sim**
```bash
source /opt/ros/humble/setup.bash
python Simulation/CARLA/scripts/mining_sim.py --num_truck 5
```

**Terminal 3 — Autonomy demo:**
```bash
source /opt/ros/humble/setup.bash
cd Demos/CARLA_mining
python CARLA_mining_demo.py \
  --contour_model    /path/to/contour.pth \
  --object_seg_model /path/to/objectseg.pth 

```

Press **ESC** in the display window to exit. See [Demos/CARLA_mining/README.md](Demos/CARLA_mining/README.md) for full pipeline and display layout details.

---

