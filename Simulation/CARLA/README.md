# CARLA Mining Simulation

`scripts/mining_sim.py` — interactive CARLA simulator for collecting off-road elevation datasets from a mining environment.

## Overview

Spawns a mining truck ego vehicle in the `Mine_01` CARLA map with a full sensor suite.  The script doubles as a manual control client and a dataset recorder: press **E** to toggle paired RGB + ground-truth elevation image recording at 2 Hz.

A ROS 2 bridge is included so an external Ackermann controller can drive the ego vehicle.

## Requirements

| Dependency | Notes |
|---|---|
| CARLA server | `Mine_01` map must be loaded |
| Python CARLA client library | must be importable |
| ROS 2 (Humble) | `rclpy`, `ackermann_msgs`, `std_msgs` |
| `pygame` | display and keyboard input |
| `numpy`, `opencv-python` | array and image ops |

## Usage

```bash
python scripts/mining_sim.py [options]
```

| Argument | Default | Description |
|---|---|---|
| `--host H` | `127.0.0.1` | CARLA server IP |
| `-p PORT` | `2000` | CARLA server TCP port |
| `--res WxH` | `1920x1080` | Display resolution |
| `--sync` | off | Synchronous simulation mode (recommended for recording) |
| `-a` / `--autopilot` | off | Start with CARLA autopilot enabled |
| `--gamma G` | `1.0` | Camera gamma correction |
| `--lead_truck true/false` | `false` | Spawn a lead mining truck 15 m ahead of the ego |
| `--num_trucks N` | `0` | Spawn N additional NPC mining trucks at random map locations with autopilot |

### Example

```bash
# Sync mode with a lead truck and 3 additional NPC trucks
python scripts/mining_sim.py --sync --lead_truck true --num_trucks 3
```

## Keyboard Controls

| Key | Action |
|---|---|
| W / Up | Throttle |
| S / Down | Brake |
| A / D | Steer left / right |
| Q | Toggle reverse |
| Space | Hand-brake |
| P | Toggle autopilot |
| F | Toggle Ackermann controller (reads from ROS 2) |
| M | Toggle manual transmission |
| , / . | Gear down / up |
| Ctrl+W | Constant velocity mode (60 km/h) |
| E | **Toggle elevation dataset recording** |
| R | Toggle RGB image recording to `data/_rgb/` |
| Ctrl+R | Toggle CARLA simulation recording |
| TAB | Cycle camera position |
| ` / N / 1–9 | Cycle / select sensor view |
| C / Shift+C | Cycle weather presets |
| G | Toggle radar visualisation |
| L / Shift+L | Cycle lights / high beam |
| Z / X | Left / right blinker |
| T | Toggle vehicle telemetry |
| V / B | Select / load map layer |
| F1 | Toggle HUD |
| H / ? | Toggle help overlay |
| Backspace | Respawn vehicle |
| ESC / Ctrl+Q | Quit |

## ROS 2 Interface

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/carla/ego_vehicle/ackermann_cmd` | `ackermann_msgs/AckermannDrive` | Subscribe | Drive commands forwarded when Ackermann mode is active (F key) |
| `/carla/ego_vehicle/speed` | `std_msgs/Float32` | Publish | Ego vehicle speed in m/s, published every tick |

`RMW_IMPLEMENTATION` is forced to `rmw_fastrtps_cpp` on startup to avoid CycloneDDS conflicts with Isaac Sim.

## Dataset Recording (E key)

Recordings are saved to `data/_elevation_dataset/` relative to `Simulation/CARLA/`.  Frames are saved at **2 Hz** whenever the LiDAR and a recent RGB frame are within 2 simulation frames of each other.  Frame indices are monotonically increasing across sessions.

### Output layout

```
data/_elevation_dataset/
├── images/                   640×320 BGR PNGs — RGB frames
├── gt_lidar_elevations/      640×320 grayscale PNGs — LiDAR-projected elevation (see encoding)
├── gt_depth_elevations/      640×320 grayscale PNGs — depth-camera-derived elevation
├── gt_bev_lidar_elevations/  600×600 grayscale PNGs — BEV LiDAR elevation (60 m × 60 m, 10 px/m)
├── gt_bev_depth_elevations/  600×600 grayscale PNGs — BEV depth-derived elevation
└── camera_params/
    └── camera_params.npy     float32 (12,) — flattened 3×4 projection matrix P = K @ [R|t]
                              shared by all frames; written once on first recording toggle
```

### Elevation PNG encoding

| Pixel value | Meaning |
|---|---|
| 0 | No data (NaN / out-of-range) |
| 96–255 | Valid elevation bin, linearly scaled |

Elevation is quantised into **40 bins** spanning **−0.5 m to +1.5 m** in 5 cm steps.  Bin index `b ∈ [0, 39]` is mapped to grayscale value `round(b * 159 / 39) + 96`, so that bin 0 → 96 and bin 39 → 255.  The reserved value 0 unambiguously marks pixels with no LiDAR/depth coverage.

To decode back to a bin index: `b = round((p − 96) × 39 / 159)` where `p` is the uint8 pixel value.  See `LoadDataElevation._load()` in `Models/data_utils/load_data_elevation.py`.

### Camera parameters

`camera_params.npy` stores the **3×4 projection matrix** `P = K @ [R|t]` at the **640×320** network resolution:

- **K** — intrinsics computed from the CARLA camera FOV at 640×320
- **R** — rotation from CARLA LiDAR frame (x=fwd, y=left, z=up) to OpenCV camera frame (x=right, y=down, z=fwd):
  ```
  R = [[ 0, -1,  0],
       [ 0,  0, -1],
       [ 1,  0,  0]]
  ```
- **t** = [0, 0, 0] (camera and LiDAR are co-mounted)

### BEV grid parameters

| Parameter | Value |
|---|---|
| Forward range | 60 m |
| Lateral range | ±30 m (60 m total) |
| Resolution | 10 px/m |
| Grid size | 600 × 600 px |
| Origin | ego vehicle position; near edge at bottom |

BEV lidar images are horizontally flipped before saving so that left in the scene appears on the left side of the image.

## Sensors

The following sensors are attached to the ego vehicle:

| Sensor | Purpose |
|---|---|
| RGB camera (display) | Live viewer; optional `_rgb/` recording via R key |
| RGB camera (dataset) | Co-mounted with LiDAR; feeds `images/` |
| Depth camera (dataset) | Same mount/FOV; feeds `gt_depth_elevations/` |
| LiDAR ray-cast (50 m) | Feeds all `gt_lidar_*` outputs |
| Collision sensor | HUD collision history |
| Lane invasion sensor | HUD notifications |
| GNSS | HUD latitude/longitude |
| IMU | HUD compass, accelerometer, gyroscope |
| Radar (optional) | Toggled with G; visualised in-world |
