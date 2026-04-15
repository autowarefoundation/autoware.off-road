#!/usr/bin/env python3

import os
# Must be set before rclpy is imported or initialised
os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import ctypes
from ctypes import c_double, c_float, c_int, c_void_p, c_ubyte, POINTER, Structure
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import re

# Path setup to import model inference classes from the Models directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Models is located at REPO_ROOT/Models/
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
MODELS_DIR = os.path.join(REPO_ROOT, "Models")

# Insert both so that internal imports like 'from model_components...' work
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, MODELS_DIR)

try:
    from Models.inference.auto_speed_infer import AutoSpeedNetworkInfer, AutoSpeedONNXInfer
    from Models.inference.freespace_contour_infer import FreespaceContourNetworkInfer
    from Models.inference.object_seg_infer import ObjectSegNetworkInfer
    print("Successfully imported model inference classes.")
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)

# Transparent yellow overlay: BGR (0, 255, 255), alpha=0.45
_VEHICLE_OVERLAY_BGR = np.array([0, 255, 255], dtype=np.float32)
_VEHICLE_OVERLAY_ALPHA = 0.45

# C-compatible structure mirroring GapFollowerParams in C++
class GapFollowerParams(Structure):
    _fields_ = [
        ("bev_resolution", ctypes.c_double),
        ("bev_ego_x", ctypes.c_int),
        ("bev_ego_y", ctypes.c_int),
        ("raycast_angle_min", ctypes.c_double),
        ("raycast_angle_max", ctypes.c_double),
        ("raycast_angle_increment", ctypes.c_double),
        ("raycast_max_range", ctypes.c_double),
        ("min_gap_size", ctypes.c_double),
        ("range_thresh", ctypes.c_double),
        ("goal_angle", ctypes.c_double),
    ]

class CARLAMiningDemoNode(Node):
    def __init__(self, contour_model_path, speed_model_path, object_seg_model_path='', lead_truck=False):
        super().__init__('carla_mining_demo')
        self.lead_truck_mode = lead_truck
        self.lead_truck_speed = 0.0  # Speed received from lead mining truck

        self.get_logger().info('Loading foundation models...')

        # Initialize Freespace Contour Inference
        # Expects the direct path to the .pt/pth file
        self.contour_infer = FreespaceContourNetworkInfer(checkpoint_path=contour_model_path)

        # Initialize Auto Speed Inference (optional)
        self.speed_infer = None
        if speed_model_path:
            if speed_model_path.endswith('.onnx'):
                self.get_logger().info(f'Loading AutoSpeed ONNX model: {speed_model_path}')
                self.speed_infer = AutoSpeedONNXInfer(onnx_path=speed_model_path)
            else:
                self.get_logger().info(f'Loading AutoSpeed PyTorch model: {speed_model_path}')
                self.speed_infer = AutoSpeedNetworkInfer(checkpoint_path=speed_model_path)

        # Initialize ObjectSeg Inference (optional — truck overlay)
        self.obj_seg_infer = None
        if object_seg_model_path:
            self.get_logger().info(f'Loading ObjectSeg model: {object_seg_model_path}')
            self.obj_seg_infer = ObjectSegNetworkInfer(checkpoint_path=object_seg_model_path)

        self.get_logger().info('Models loaded successfully.')

        # Initialize C++ Gap Follower via ctypes
        self.init_cpp_gap_follower()
        self.init_controllers()
        
        # QoS Profile for high-bandwidth image data
        # Using RELIABLE ensures DDS recovers dropped UDP fragments, 
        # which is critical for larger Image messages on local networks.
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=5
        )

        # Dynamic Subscription Setup
        self.last_msg_time = 0.0          # tracks when we last received a frame

        # Subscribe directly to the ego vehicle camera published by mining_sim.py
        _ego_topic = '/carla/ego_vehicle/rgb/image'
        self.current_topic = _ego_topic
        self.subscription = self.create_subscription(
            Image, _ego_topic, self.image_callback, self.qos_profile)
        self.get_logger().info(f'Subscribed to camera topic: {_ego_topic}')

        # Slow timer (0.2 Hz = every 5s) to handle vehicle respawns / topic changes
        self.discovery_timer = self.create_timer(5.0, self.discover_topics)
            
        # [DISABLED] Publishers
        # self.publisher = self.create_publisher(Image, '/demo/output_image', self.qos_profile)

        # AutoSpeed class colors
        self.speed_color_map = {
            1: (0, 0, 255),    # red
            2: (0, 255, 255),  # yellow
            3: (255, 255, 0)   # cyan
        }
        # Threading for async processing
        self.thread_executor = ThreadPoolExecutor(max_workers=8)
        # Dedicated pool for parallel model inference (one thread per model)
        self._infer_executor = ThreadPoolExecutor(max_workers=3)
        self.processing_lock = threading.Lock()
        self.is_processing = False

        # Producer-consumer display buffer.
        self.ready_frame = None
        self.frame_lock = threading.Lock()
        
        self.contour_history = deque(maxlen=5) # Smooth over last 5 frames
        self.contour_lock = threading.Lock()
        self._frame_count = 0  # for throttling logs
        self._prev_mask_area = 0  # for truck-leaving detection
        self.autonomous = True  # toggled by Space in the display window

        # Pre-compute BEV projection constants so process_frame doesn't recompute every frame
        _res = 20  # px/m
        self._bev_W      = int(40  * _res)   # 800 px
        self._bev_H_draw = int(100 * _res)   # 2000 px (full warp canvas)
        self._bev_H_crop = int(50  * _res)   # 1000 px (displayed near range)
        _K = np.array([[960.0, 0, 960.0], [0, 960.0, 540.0], [0, 0, 1.0]])
        _Rt = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 3.25], [1.0, 0.0, -3.2]])
        _scale = np.array([[640/1920.0, 0, 0], [0, 360/1080.0, 0], [0, 0, 1]])
        _H_world = _scale @ (_K @ _Rt)
        _M_td2w = np.array([[0, -1.0/_res, 105.0], [1.0/_res, 0, -20.0], [0, 0, 1]])
        self._M_td2img = _H_world @ _M_td2w
        self._M_img2td = np.linalg.inv(self._M_td2img)
        # Crop-shifted warp: maps output (u, v) → full-BEV (u, v + H_crop), letting us
        # warp only the near 50m directly into a (W, H_crop) canvas instead of (W, H_draw)
        _shift = np.array([[1, 0, 0], [0, 1, float(self._bev_H_crop)], [0, 0, 1]])
        self._M_td2img_crop = self._M_td2img @ _shift
        # Pre-computed radial angles for contour projection (37 bins)
        self._bev_angles = np.linspace(np.pi, 0, 37)

    def init_cpp_gap_follower(self):
        """Load libGapFollower.so and setup function signatures."""
        lib_path = os.path.join(REPO_ROOT, "Modules/GapFollower/build/libGapFollower.so")
        if not os.path.exists(lib_path):
            self.get_logger().error(f"GapFollower library not found at {lib_path}. Build it first!")
            self.lib_gf = None
            return

        self.lib_gf = ctypes.CDLL(lib_path)

        # void* create_gap_follower(const GapFollowerParams* params)
        self.lib_gf.create_gap_follower.argtypes = [POINTER(GapFollowerParams)]
        self.lib_gf.create_gap_follower.restype = c_void_p

        # void destroy_gap_follower(void* handle)
        self.lib_gf.destroy_gap_follower.argtypes = [c_void_p]
        self.lib_gf.destroy_gap_follower.restype = None

        # void process_bev_contour(void* handle, unsigned char* data, int rows, int cols, 
        #                         int* gap_idx1, int* gap_idx2,
        #                         int* all_gaps, int* num_gaps, int max_gaps)
        self.lib_gf.process_bev_contour.argtypes = [
            c_void_p, POINTER(c_ubyte), c_int, c_int, 
            POINTER(c_int), POINTER(c_int),
            POINTER(c_int), POINTER(c_int), c_int
        ]
        self.lib_gf.process_bev_contour.restype = None

        # Instantiate C++ class
        params = GapFollowerParams()
        params.bev_resolution = 0.05
        params.bev_ego_x = 400
        params.bev_ego_y = 2000
        params.raycast_angle_min = -np.pi / 2
        params.raycast_angle_max = np.pi / 2
        params.raycast_angle_increment = np.pi / 180.0
        params.raycast_max_range = 50.0
        params.min_gap_size = 5.0
        params.range_thresh = 25.0  # Set as requested per user
        params.goal_angle = 0.0

        self.gf_handle = self.lib_gf.create_gap_follower(ctypes.byref(params))
        self.get_logger().info("C++ GapFollower module loaded and initialized.")
        
    def init_controllers(self):
        """Load PI_Controller and SteeringController lib and set up ctypes bindings"""
        pi_lib_path = os.path.join(REPO_ROOT, "Modules/Control/Longitudinal/PI_Controller/build/libpi_controller_lib.so")
        self.lib_pi = ctypes.CDLL(pi_lib_path)
        self.lib_pi.create_pi_controller.argtypes = [c_double, c_double, c_double]
        self.lib_pi.create_pi_controller.restype = c_void_p
        self.lib_pi.pi_compute_effort.argtypes = [c_void_p, c_double, c_double]
        self.lib_pi.pi_compute_effort.restype = c_double
        # K_p, K_i, K_d for speed
        self.pi_handle = self.lib_pi.create_pi_controller(2.0, 0.001, 0.05)

        steer_lib_path = os.path.join(REPO_ROOT, "Modules/Control/Steering/SteeringController/build/libsteering_controller_lib.so")
        self.lib_steer = ctypes.CDLL(steer_lib_path)
        self.lib_steer.create_steering_controller.argtypes = [c_double, c_double, c_double, c_double]
        self.lib_steer.create_steering_controller.restype = c_void_p
        self.lib_steer.steering_compute.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
        self.lib_steer.steering_compute.restype = c_double
        # wheelbase=3.0, K_p, K_i, K_d for steering
        self.steer_handle = self.lib_steer.create_steering_controller(3.0, 1.3, 0.05, 0.2)

        # Topic for AckermannDrive
        self.ackermann_pub = self.create_publisher(AckermannDrive, '/carla/ego_vehicle/ackermann_cmd', 10)
        
        # Vehicle speed subscription
        self.current_speed = 0.0
        self.speed_sub = self.create_subscription(Float32, '/carla/ego_vehicle/speed', self.speed_callback, 10)

        # Lead truck speed subscription (enabled via --lead_truck true)
        if self.lead_truck_mode:
            self.lead_truck_speed_sub = self.create_subscription(
                Float32, '/carla/lead_truck/speed', self.lead_truck_speed_callback, 10)
            self.get_logger().info("Lead truck mode ENABLED — subscribing to /carla/lead_truck/speed")
        else:
            self.lead_truck_speed_sub = None

        self.get_logger().info("C++ Controllers initialized and Ackermann publisher created.")

        # Note: cv2.imshow must run in the main thread.
        # We will poll `ready_frame` from the main loop in `main()`.

    def speed_callback(self, msg):
        self.current_speed = msg.data

    def lead_truck_speed_callback(self, msg):
        self.lead_truck_speed = msg.data

    def discover_topics(self):
        """Scans for /carla/vehicle{N}/rgb{M}/image.
        When already subscribed and receiving frames, this call is a no-op to
        avoid contending on ROS2 middleware locks with the publisher.
        """
        now = time.time()
        # Skip expensive API call if we are actively receiving frames (within last 2s)
        if self.current_topic and (now - self.last_msg_time) < 2.0:
            return

        topic_list = self.get_topic_names_and_types()
        pattern = re.compile(r'/carla/vehicle(\d+)/rgb\d+/image')
        ego_topic = '/carla/ego_vehicle/rgb/image'

        best_topic = None
        max_vid = -1

        for topic_name, _ in topic_list:
            if topic_name == ego_topic:
                # ego vehicle camera is the default; only switch if a higher-id vehicle topic exists
                if best_topic is None:
                    best_topic = ego_topic
                continue
            match = pattern.match(topic_name)
            if match:
                vid = int(match.group(1))
                if vid > max_vid:
                    max_vid = vid
                    best_topic = topic_name
        
        if best_topic and best_topic != self.current_topic:
            self.get_logger().info(f'New camera topic discovered: {best_topic}. Subscribing...')
            
            # Remove old subscription if exists
            if self.subscription:
                self.destroy_subscription(self.subscription)
            
            self.current_topic = best_topic
            self.subscription = self.create_subscription(
                Image,
                self.current_topic,
                self.image_callback,
                self.qos_profile)

    def image_callback(self, msg):
        """High-rate callback that downsamples and offloads."""
        self.last_msg_time = time.time()  # track activity for smart discovery
        if self.is_processing:
            return
            
        with self.processing_lock:
            self.is_processing = True

        try:
            # 1. Decode and Downsample IMMEDIATELY (takes ~10-15ms)
            # This allows the ROS2 executor thread to handle the next 1080p frame quickly.
            channels = 3
            if msg.encoding in ['bgra8', 'rgba8']:
                channels = 4
            
            raw_frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            
            # We work in BGR for CV2
            if msg.encoding in ['rgb8', 'rgba8']:
                frame_bgr_full = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR if channels==3 else cv2.COLOR_RGBA2BGR)
            elif msg.encoding in ['bgr8', 'bgra8']:
                frame_bgr_full = raw_frame if channels==3 else cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr_full = raw_frame
            
            # mining_sim.py publishes at 640×360 already; only resize if receiving full-res
            if msg.width == 640 and msg.height == 360:
                frame_bgr_360 = frame_bgr_full
            else:
                frame_bgr_360 = cv2.resize(frame_bgr_full, (640, 360), interpolation=cv2.INTER_LINEAR)
            
            # Offload lightweight 0.7MB buffer instead of 6MB 1080p buffer
            self.thread_executor.submit(self.process_frame, msg.header, frame_bgr_360)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")
            with self.processing_lock:
                self.is_processing = False

    def process_frame(self, header, frame_bgr_360):
        """Asynchronous low-resolution processing pipeline."""
        t_start = time.time()
        try:
            # frame_bgr_360 is (360, 640, 3) BGR
            frame_rgb_360 = cv2.cvtColor(frame_bgr_360, cv2.COLOR_BGR2RGB)
            
            # 1. Prepare Inputs for parallel inference
            # Contour (640x320)
            contour_input_rgb = cv2.resize(frame_rgb_360, (640, 320), interpolation=cv2.INTER_LINEAR)
            pil_contour = PILImage.fromarray(contour_input_rgb)
            
            t_prep = time.time()

            # 2. Dispatch contour and objectseg concurrently — both share the
            #    SceneSegNetwork backbone so they benefit from the same CUDA stream.
            #    PyTorch releases the GIL during CUDA ops so threads truly overlap.
            def _timed(fn, *args):
                t0 = time.time()
                result = fn(*args)
                return result, (time.time() - t0) * 1000.0

            f_contour = self._infer_executor.submit(_timed, self.contour_infer.inference, pil_contour)

            f_obj = None
            if self.obj_seg_infer is not None:
                f_obj = self._infer_executor.submit(_timed, self.obj_seg_infer.inference, pil_contour)

            # Collect results (blocks until each finishes)
            contour_indices,  t_contour_ms = f_contour.result()
            speed_predictions = []   # AutoSpeed not used
            obj_pred,         t_obj_ms     = f_obj.result() if f_obj is not None else (None, 0.0)

            # Truck pixel area (model res 320×640) — proxy for distance; 0 if no model or no detection
            vehicle_mask_area = int(np.sum(obj_pred == 3)) if obj_pred is not None else 0

            # Temporal Smoothing (Moving Average)
            with self.contour_lock:
                self.contour_history.append(contour_indices)
                smoothed_contour = np.mean(self.contour_history, axis=0)

                # Spatial Smoothing (Neighbor Average)
                padded_contour = np.pad(smoothed_contour, 1, mode='edge')
                window = np.ones(3) / 3.0
                smoothed_contour = np.convolve(padded_contour, window, mode='valid')

            t_inf = time.time()

            # 3. Create clean visualization for front-view with RED contour
            frame_bgr_360_view = frame_bgr_360.copy()

            # ObjectSeg: transparent yellow overlay on detected vehicles (class 3)
            if obj_pred is not None:
                vehicle_mask = (obj_pred == 3)                          # mining truck pixels
                if vehicle_mask.any():
                    # Scale mask from model res (320) to display res (360)
                    vehicle_mask_360 = cv2.resize(
                        vehicle_mask.astype(np.uint8), (640, 360),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    frame_bgr_360_view[vehicle_mask_360] = (
                        frame_bgr_360_view[vehicle_mask_360] * (1 - _VEHICLE_OVERLAY_ALPHA)
                        + _VEHICLE_OVERLAY_BGR * _VEHICLE_OVERLAY_ALPHA
                    ).astype(np.uint8)

            self.draw_contour(frame_bgr_360_view, smoothed_contour, color=(0, 0, 255), thickness=1)
            
            # Visualization for AutoSpeed on front view
            for pred in speed_predictions:
                x1, y1, x2, y2, conf, cls = pred
                disp_x1, disp_y1 = int(x1), int(y1 * (360.0 / 640.0))
                disp_x2, disp_y2 = int(x2), int(y2 * (360.0 / 640.0))
                color = self.speed_color_map.get(int(cls), (255, 255, 255))
                cv2.rectangle(frame_bgr_360_view, (disp_x1, disp_y1), (disp_x2, disp_y2), color, 2)
                cv2.putText(frame_bgr_360_view, f'cls:{int(cls)}', (disp_x1, disp_y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            t_draw = time.time()
            
            # 4. Homography and Top-Down View Refinement
            try:
                # Use pre-computed BEV constants (cached in __init__)
                W_td      = self._bev_W
                H_td_draw = self._bev_H_draw
                H_td_crop = self._bev_H_crop
                M_td2img  = self._M_td2img
                M_img2td  = self._M_img2td

                # Warp only the near 50m into a (W_td x H_td_crop) canvas using the
                # crop-shifted matrix — 2x fewer pixels than the full 100m canvas.
                topdown_clean = cv2.warpPerspective(
                    frame_bgr_360, self._M_td2img_crop,
                    (W_td, H_td_crop),
                    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
                
                # 5. Point-based BEV contour mask
                angles = self._bev_angles   # pre-computed in __init__
                img_pts = []
                for idx, angle in zip(smoothed_contour, angles):
                    dist = idx * 10
                    r = (319 - dist * np.sin(angle)) * (360.0/320.0)
                    c = 320 + dist * np.cos(angle)
                    img_pts.append([c, r, 1.0])

                # topdown_clean is H_td_crop tall; mask must match
                topdown_mask = np.zeros((H_td_crop, W_td), dtype=np.uint8)
                bev_pts_vis = []
                bev_pts_nav = []  # Full BEV coords for bev_fill (C++ gap follower)
                for p_img in img_pts:
                    p_td = M_img2td @ p_img
                    u, v = p_td[0]/p_td[2], p_td[1]/p_td[2]

                    v_nav = v
                    if v_nav > H_td_draw: v_nav = float(H_td_draw)
                    bev_pts_nav.append((int(u), int(v_nav)))

                    # Shift into crop-canvas space (canvas row 0 = full-BEV row H_td_crop)
                    v_vis = v - H_td_crop
                    if v_vis > H_td_crop: v_vis = 0.0  # snap to top
                    bev_pts_vis.append((int(u), int(v_vis)))

                # Draw red contour on mask using VIS points (crop-canvas coords)
                for i in range(len(bev_pts_vis)-1):
                    p1, p2 = bev_pts_vis[i], bev_pts_vis[i+1]
                    if (-100 < p1[0] < W_td + 100) and (-100 < p1[1] < H_td_crop + 100):
                        cv2.line(topdown_mask, p1, p2, 255, thickness=2)
                        cv2.circle(topdown_mask, p1, 3, 255, -1)
                cv2.circle(topdown_mask, bev_pts_vis[-1], 3, 255, -1)
                
                # Overlay RED contour on CLEAN 100m bev image
                topdown_clean[topdown_mask > 125] = [0, 0, 255]
                
                # 6. Gap Follower Logic (using C++ module)
                bev_fill = np.zeros((H_td_draw, W_td), dtype=np.uint8)
                if len(bev_pts_nav) > 2:
                    for i in range(len(bev_pts_nav)-1):
                        p1, p2 = bev_pts_nav[i], bev_pts_nav[i+1]
                        # check the distance value > bev height
                        if p1[1] >= H_td_draw or p2[1] >= H_td_draw:
                            continue
                        if (-100 < p1[0] < W_td + 100) and (-100 < p1[1] < H_td_draw + 100):
                            cv2.line(bev_fill, p1, p2, 255, thickness=2)
                            cv2.circle(bev_fill, p1, 3, 255, -1)
                    if bev_pts_nav[-1][1] < H_td_draw:
                        cv2.circle(bev_fill, bev_pts_nav[-1], 3, 255, -1)
                
                # Project truck bottom-center from image space → BEV for direct truck following.
                # The lowest mask row is the truck's ground contact point (closest to ego).
                truck_bev_pt = None  # (center_px, center_py) in full-BEV space, or None
                if vehicle_mask_area > 2500 and obj_pred is not None:
                    mask_rows, mask_cols = np.where(obj_pred == 3)
                    if len(mask_rows) > 0:
                        bottom_row  = int(mask_rows.max())
                        center_col  = float(mask_cols.mean())
                        # Scale from model res (320×640) to homography res (640×360)
                        r_360 = bottom_row * (360.0 / 320.0)
                        p_img = np.array([center_col, r_360, 1.0])
                        p_td  = M_img2td @ p_img
                        if p_td[2] > 1e-3:
                            tx = int(p_td[0] / p_td[2])
                            ty = int(p_td[1] / p_td[2])
                            if 0 <= tx < W_td and 0 < ty <= H_td_draw:
                                truck_bev_pt = (tx, ty)

                # Replace Python logic with C++ processing
                t_gap_start = time.time()
                t_traj_ms = 0.0
                if self.lib_gf and self.gf_handle:
                    idx1 = c_int(0)
                    idx2 = c_int(0)
                    max_gaps = 50
                    all_gaps = (c_int * int(max_gaps * 6))()
                    num_gaps = c_int(0)

                    # Ensure bev_fill is contiguous for ctypes
                    bev_data = np.ascontiguousarray(bev_fill, dtype=np.uint8)

                    self.lib_gf.process_bev_contour(
                        self.gf_handle, 
                        bev_data.ctypes.data_as(POINTER(c_ubyte)), 
                        H_td_draw, W_td,
                        ctypes.byref(idx1), ctypes.byref(idx2),
                        all_gaps, ctypes.byref(num_gaps), max_gaps
                    )
                    
                    t_gap_ms = (time.time() - t_gap_start) * 1000.0

                    # Select the largest gap by endpoint-to-endpoint distance
                    best_i = 0
                    best_width = -1.0
                    for i in range(num_gaps.value):
                        _px1 = all_gaps[i * 6 + 2]
                        _py1 = all_gaps[i * 6 + 3]
                        _px2 = all_gaps[i * 6 + 4]
                        _py2 = all_gaps[i * 6 + 5]
                        w = np.sqrt((_px2 - _px1) ** 2 + (_py2 - _py1) ** 2)
                        if w > best_width:
                            best_width = w
                            best_i = i
                    gap_idx1 = all_gaps[best_i * 6 + 0]
                    gap_idx2 = all_gaps[best_i * 6 + 1]

                    if self._frame_count % 30 == 0:
                        self.get_logger().info(f"Largest Gap: idx1={gap_idx1}, idx2={gap_idx2}, width={best_width:.1f}px, Total Gaps={num_gaps.value}")

                    # Drawing logic for ALL gaps
                    for i in range(num_gaps.value):
                        g1 = all_gaps[i * 6 + 0]
                        g2 = all_gaps[i * 6 + 1]
                        px1 = all_gaps[i * 6 + 2]
                        py1 = all_gaps[i * 6 + 3]
                        px2 = all_gaps[i * 6 + 4]
                        py2 = all_gaps[i * 6 + 5]

                        is_target = (g1 == gap_idx1 and g2 == gap_idx2)
                        color = (0, 255, 0) if is_target else (0, 180, 0)
                        thick = 2 if is_target else 1

                        # Draw on BEV view — py is in full-BEV space; subtract H_td_crop for canvas
                        py1_c = py1 - H_td_crop
                        py2_c = py2 - H_td_crop
                        cv2.line(topdown_clean, (px1, py1_c), (px2, py2_c), color, thick)

                        # Draw on front view by projecting full-BEV coords via M_td2img
                        p1_td = np.array([px1, py1, 1.0])
                        p1_img = M_td2img @ p1_td
                        c_f1, r_f1 = int(p1_img[0]/p1_img[2]), int(p1_img[1]/p1_img[2])

                        p2_td = np.array([px2, py2, 1.0])
                        p2_img = M_td2img @ p2_td
                        c_f2, r_f2 = int(p2_img[0]/p2_img[2]), int(p2_img[1]/p2_img[2])

                        if 0 <= r_f1 < 360 and 0 <= c_f1 < 640 and 0 <= r_f2 < 360 and 0 <= c_f2 < 640:
                            cv2.line(frame_bgr_360_view, (c_f1, r_f1), (c_f2, r_f2), color, thick)

                        if is_target:
                            t_traj_start = time.time()
                            # Default goal: gap center
                            center_px = int((px1 + px2) / 2)
                            center_py = int((py1 + py2) / 2)  # full-BEV
                            # Override with truck ground point when detected and close
                            if truck_bev_pt is not None:
                                center_px, center_py = truck_bev_pt

                            ego_px, ego_py = W_td // 2, H_td_draw  # full-BEV ego coords
                            D = ego_py - center_py  # distance in full-BEV rows

                            # Compute control commands (full-BEV coords)
                            X_fwd = D / 20.0  # 20 px/m resolution
                            Y_left = (ego_px - center_px) / 20.0
                            cte = Y_left
                            yaw_error = np.arctan2(Y_left, max(X_fwd, 0.1))

                            # Target speed: lead truck overrides; otherwise reduced by truck proximity
                            if self.lead_truck_mode:
                                target_speed = self.lead_truck_speed
                            else:
                                _MAX_MS     = 50.0 / 3.6   # 50 km/h — no truck detected
                                _TRUCK_MS   = 45.0 / 3.6   # 45 km/h — truck detected
                                _MIN_MS     = 3.0           # minimum speed when truck is very close
                                _AREA_FAR   = 1500          # px — truck visible but distant
                                _AREA_CLOSE = 4000          # px — reach min speed
                                area_delta  = vehicle_mask_area - self._prev_mask_area
                                if vehicle_mask_area == 0:
                                    target_speed = _MAX_MS
                                elif vehicle_mask_area <= _AREA_FAR:
                                    target_speed = _TRUCK_MS
                                elif area_delta < -150:
                                    # Truck leaving — use speed for an area 2000 px smaller to accelerate earlier
                                    effective_area = max(vehicle_mask_area - 2000, _AREA_FAR)
                                    t = min((effective_area - _AREA_FAR) /
                                            float(_AREA_CLOSE - _AREA_FAR), 1.0)
                                    target_speed = _MIN_MS + (1.0 - t) * (_TRUCK_MS - _MIN_MS)
                                else:
                                    t = min((vehicle_mask_area - _AREA_FAR) /
                                            float(_AREA_CLOSE - _AREA_FAR), 1.0)
                                    target_speed = _MIN_MS + (1.0 - t) * (_TRUCK_MS - _MIN_MS)
                            self._prev_mask_area = vehicle_mask_area

                            steer_angle = self.lib_steer.steering_compute(self.steer_handle, cte, yaw_error, target_speed, 0.0)
                            speed_effort = self.lib_pi.pi_compute_effort(self.pi_handle, float(self.current_speed), target_speed)

                            # Wider clamp when accelerating after truck leaves
                            accel_cap = 8.0 if (vehicle_mask_area < self._prev_mask_area) else 4.0
                            speed_effort = max(min(speed_effort, accel_cap), -6.0)

                            msg = AckermannDrive()
                            if self.autonomous:
                                msg.steering_angle = -float(steer_angle)
                                msg.speed = target_speed
                                msg.acceleration = float(speed_effort)
                            else:
                                msg.steering_angle = 0.0
                                msg.speed = 0.0
                                msg.acceleration = -3.0  # brake
                            self.ackermann_pub.publish(msg)

                            # Bezier in crop-canvas coords (subtract H_td_crop from all y)
                            ego_py_c   = H_td_crop           # canvas row for ego (bottom of canvas)
                            center_py_c = center_py - H_td_crop
                            p0 = np.array([ego_px,    ego_py_c])
                            p1 = np.array([ego_px,    ego_py_c    - D * 0.4])
                            p2 = np.array([center_px, center_py_c + D * 0.4])
                            p3 = np.array([center_px, center_py_c])

                            t = np.linspace(0, 1, 20)[:, None]
                            curve = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
                            curve = curve.astype(np.int32)

                            if False:  # trajectory drawings temporarily hidden
                                # Draw on BEV view (Cyan) — curve is already in canvas coords
                                cv2.polylines(topdown_clean, [curve], isClosed=False, color=(255, 255, 0), thickness=3)

                                # Draw on front view — convert canvas coords back to full-BEV for M_td2img
                                curve_bev = curve.copy().astype(np.float64)
                                curve_bev[:, 1] += H_td_crop  # canvas → full-BEV y
                                curve_td = np.hstack((curve_bev, np.ones((len(curve_bev), 1))))
                                p_img_all = (M_td2img @ curve_td.T).T

                                # Filter points that are actually in front of the camera
                                valid_mask = p_img_all[:, 2] > 1e-3
                                if np.any(valid_mask):
                                    curve_img = p_img_all[valid_mask]
                                    curve_img = curve_img[:, :2] / curve_img[:, 2:]
                                    cv2.polylines(frame_bgr_360_view, [curve_img.astype(np.int32)], isClosed=False, color=(255, 255, 0), thickness=3)
                            t_traj_ms = (time.time() - t_traj_start) * 1000.0

                # Resize and output stack — topdown_clean is already the near-50m crop
                td_resized = cv2.resize(topdown_clean, (int(W_td * (540/H_td_crop)), 540), interpolation=cv2.INTER_NEAREST)
                front_resized = cv2.resize(frame_bgr_360_view, (960, 540), interpolation=cv2.INTER_LINEAR)
                frame_out = np.hstack((front_resized, td_resized))
            except Exception as e:
                self.get_logger().error(f"IPM Warning: {e}")
                frame_out = cv2.resize(frame_bgr_360, (960, 540), interpolation=cv2.INTER_LINEAR)

            # Write to producer-consumer buffer; executor timer will display
            with self.frame_lock:
                self.ready_frame = frame_out

            t_end = time.time()

            self.get_logger().info(
                f"FPS: {1.0/(t_end - t_start):.1f} | Total: {(t_end - t_start)*1000:.1f}ms | "
                f"Contour: {t_contour_ms:.1f}ms | ObjSeg: {t_obj_ms:.1f}ms | "
                f"Gap: {t_gap_ms:.1f}ms | Traj: {t_traj_ms:.1f}ms | "
                f"Draw: {(t_draw-t_inf)*1000:.1f}ms"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in process_frame: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self._frame_count += 1
            with self.processing_lock:
                self.is_processing = False


    def draw_contour(self, image, contour_indices, color=(0, 255, 0), thickness=1):
        """Draws the drivable contour points scaled to 640x360."""
        # image is 640x360
        num_rays = len(contour_indices)
        angles = np.linspace(np.pi, 0, num_rays)
        
        # Base parameters for the rays from 640x320 context
        start_r_base = 319 # Bottom of the 320 height
        start_c_base = 320 # Center of the 640 width
        ray_slice_dist_base = 10
        
        # Scale Y from 320 base to 360 display
        scale_y = 360.0 / 320.0
        
        prev_pt = None
        
        for idx, angle in zip(contour_indices, angles):
            dist = idx * ray_slice_dist_base
            r_base = start_r_base - dist * np.sin(angle)
            c_base = start_c_base + dist * np.cos(angle)
            
            final_c = int(c_base)
            final_r = int(r_base * scale_y)
            
            if 0 <= final_r < 360 and 0 <= final_c < 640:
                current_pt = (final_c, final_r)
                
                # Draw connecting line
                if prev_pt is not None:
                    cv2.line(image, prev_pt, current_pt, color, thickness=thickness)
                
                # Draw the point
                cv2.circle(image, current_pt, thickness + 1, color, -1)
                prev_pt = current_pt
            else:
                prev_pt = None

def main():
    parser = argparse.ArgumentParser(description='ROS2 Demo Node for Foundation Models in CARLA')
    parser.add_argument('--contour_model', type=str, required=True,
                        help='Path to the freespace_contour model (best.pt)')
    parser.add_argument('--speed_model', type=str, default='',
                        help='Path to the auto_speed model (best.pt or its directory) (optional)')
    parser.add_argument('--object_seg_model', type=str, default='',
                        help='Path to the ObjectSeg model checkpoint (.pth) for vehicle overlay (optional)')
    parser.add_argument('--lead_truck', type=lambda x: x.lower() == 'true', default=False,
                        help='Enable lead mining truck following mode (default: false). '
                             'Subscribes to /carla/lead_truck/speed and uses it as the target speed.')

    args = parser.parse_args()

    # Ensure paths exist
    if not os.path.exists(args.contour_model):
        print(f"Error: Contour model path not found: {args.contour_model}")
        return
    if args.speed_model and not os.path.exists(args.speed_model):
        print(f"Error: AutoSpeed model path not found: {args.speed_model}")
        return
    if args.object_seg_model and not os.path.exists(args.object_seg_model):
        print(f"Error: ObjectSeg model path not found: {args.object_seg_model}")
        return

    rclpy.init()
    node = CARLAMiningDemoNode(args.contour_model, args.speed_model,
                               object_seg_model_path=args.object_seg_model,
                               lead_truck=args.lead_truck)
    
    # Use MultiThreadedExecutor to handle dynamic topic discovery and high-rate camera input
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    
    # Spin ROS2 in a background thread so the main thread can handle OpenCV GUI
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        while rclpy.ok():
            frame = None
            with node.frame_lock:
                if node.ready_frame is not None:
                    frame = node.ready_frame
                    node.ready_frame = None
            
            if frame is not None:
                cv2.imshow("CARLA Mining Demo", frame)
            
            # cv2.waitKey handles GUI events; must be in main thread
            key = cv2.waitKey(30)
            if key == 27:   # ESC — exit
                print("ESC pressed. Exiting...")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
