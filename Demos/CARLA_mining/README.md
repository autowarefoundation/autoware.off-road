# CARLA Mining Demo

Closed-loop autonomy stack running inside the CARLA simulator on the Mine_01 map.
Subscribes to the ego vehicle's camera topic, runs perception inference, and
publishes `AckermannDrive` commands back to CARLA at ~10 Hz.

---

## Pipeline

```
CARLA camera (ROS 2 Image)
        |
        v
  [Decode + resize to 640x360]
        |
        +---> ObjectSegNetwork (640x320)
        |       Vehicle pixels (class 3) -> transparent yellow overlay
        |
        +---> FreespaceContourNetwork (640x320)
        |       37 radial bins -> temporal + spatial smoothing
        |       -> BEV contour -> GapFollower (C++)
        |           Best gap -> CTE + yaw error
        |           -> SteeringController (C++) -> steering_angle
        |
        +---> AutoSpeedNetwork (640x640)
                Detections -> target speed
                -> PI_Controller (C++) -> throttle/brake

AckermannDrive -> /carla/ego_vehicle/ackermann_cmd
```

---

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--contour_model` | yes | Path to FreespaceContour checkpoint (`.pth`) |
| `--speed_model` | no | Path to AutoSpeed checkpoint (`.pt` / `.pth` / `.onnx`) |
| `--object_seg_model` | no | Path to ObjectSeg checkpoint (`.pth`); enables vehicle overlay |
| `--lead_truck` | no | `true` / `false` — follow lead truck speed (default: `false`) |

---

## Usage

**Basic (contour + speed only):**
```bash
python CARLA_mining_demo.py \
  --contour_model /path/to/contour.pth \
  --speed_model   /path/to/speed.pth
```

**With vehicle overlay:**
```bash
python CARLA_mining_demo.py \
  --contour_model    /path/to/contour.pth \
  --speed_model      /path/to/speed.pth \
  --object_seg_model /path/to/objectseg.pth
```

**With lead-truck following:**
```bash
python CARLA_mining_demo.py \
  --contour_model    /path/to/contour.pth \
  --speed_model      /path/to/speed.pth \
  --object_seg_model /path/to/objectseg.pth \
  --lead_truck true
```

Press **ESC** in the display window to exit.

---

## ObjectSeg Vehicle Overlay

When `--object_seg_model` is provided, ObjectSegNetwork runs on every frame
using the same 640×320 input as FreespaceContourNetwork (no extra resize cost).

- **Class detected:** 3 — vehicle / mining truck
- **Overlay colour:** yellow (BGR 0, 255, 255)
- **Alpha:** 0.45 (transparent blend onto the raw camera frame)
- **Rendering order:** yellow mask → red contour line → speed detection boxes

The mask is upscaled from 320 to 360 px height using nearest-neighbour
interpolation to match the display resolution before blending.

Omitting `--object_seg_model` disables the overlay entirely with no
performance cost.

---

## Display Layout

```
[ Front camera view (960x540) | BEV top-down view (variable width x 540) ]
```

- **Front view:** raw camera with yellow vehicle overlay, red freespace contour,
  speed detection boxes, and projected gap/path lines
- **BEV view:** 50 m top-down warp with red contour, green gap indicators,
  and yellow Bézier path to the target gap centre

---

## Prerequisites

- CARLA server running with Mine_01 map loaded
- `mining_sim.py` running (spawns ego vehicle and publishes ROS 2 topics)
- GapFollower shared library built:
  ```bash
  cd Modules/GapFollower && mkdir build && cd build && cmake .. && make
  ```
- PI_Controller and SteeringController shared libraries built similarly
- ROS 2 Humble sourced, `rmw_fastrtps_cpp` available
