#!/usr/bin/env python3

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
import os
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
    print("Successfully imported model inference classes.")
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)

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
    def __init__(self, contour_model_path, speed_model_path, lead_truck=False):
        super().__init__('carla_mining_demo')
        self.lead_truck_mode = lead_truck
        self.lead_truck_speed = 0.0  # Speed received from lead mining truck
        
        self.get_logger().info('Loading foundation models...')
        
        # Initialize Freespace Contour Inference
        # Expects the direct path to the .pt/pth file
        self.contour_infer = FreespaceContourNetworkInfer(checkpoint_path=contour_model_path)
        
        # Initialize Auto Speed Inference
        if speed_model_path.endswith('.onnx'):
            self.get_logger().info(f'Loading AutoSpeed ONNX model: {speed_model_path}')
            self.speed_infer = AutoSpeedONNXInfer(onnx_path=speed_model_path)
        else:
            # AutoSpeedNetworkInfer now handles both directory (with best.pt) and direct .pt/.pth file
            self.get_logger().info(f'Loading AutoSpeed PyTorch model: {speed_model_path}')
            self.speed_infer = AutoSpeedNetworkInfer(checkpoint_path=speed_model_path)
        
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
        self.current_topic = None
        self.subscription = None
        self.last_msg_time = 0.0          # tracks when we last received a frame
        self.discovery_interval = 0       # counts ticks until next discovery attempt
        
        # Initial topic discovery (runs the expensive API call once at startup)
        self.discover_topics()
        
        # Slow timer (0.2 Hz = every 5s) to handle vehicle respawns with minimal middleware impact
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
        self.processing_lock = threading.Lock()
        self.is_processing = False

        # Producer-consumer display buffer.
        self.ready_frame = None
        self.frame_lock = threading.Lock()
        
        self.contour_history = deque(maxlen=5) # Smooth over last 5 frames
        self.contour_lock = threading.Lock()

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
        # Reduced heavily to prevent integral windup oscillation at 20fps logic loops
        self.pi_handle = self.lib_pi.create_pi_controller(0.2, 0.001, 0.05)

        steer_lib_path = os.path.join(REPO_ROOT, "Modules/Control/Steering/SteeringController/build/libsteering_controller_lib.so")
        self.lib_steer = ctypes.CDLL(steer_lib_path)
        self.lib_steer.create_steering_controller.argtypes = [c_double, c_double, c_double, c_double]
        self.lib_steer.create_steering_controller.restype = c_void_p
        self.lib_steer.steering_compute.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
        self.lib_steer.steering_compute.restype = c_double
        # wheelbase=3.0, K_p, K_i, K_d for steering
        self.steer_handle = self.lib_steer.create_steering_controller(3.0, 2.0, 0.05, 0.2)

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
        
        best_topic = None
        max_vid = -1
        
        for topic_name, _ in topic_list:
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
            
            # Resizing 1080p -> 360p is much faster than processing 1080p in models
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
            
            # Speed (640x640)
            speed_input_rgb = cv2.resize(frame_rgb_360, (640, 640), interpolation=cv2.INTER_LINEAR)
            pil_speed = PILImage.fromarray(speed_input_rgb)

            t_prep = time.time()

            # 2. Run Inference sequentially in the worker thread
            contour_indices = self.contour_infer.inference(pil_contour)
            
            # Temporal Smoothing (Moving Average)
            with self.contour_lock:
                self.contour_history.append(contour_indices)
                # axis=0 averages each ray's distance across frames
                smoothed_contour = np.mean(self.contour_history, axis=0)
                
                # Spatial Smoothing (Neighbor Average)
                # Use edge padding to avoid pulling the boundary towards 0 at the edges
                padded_contour = np.pad(smoothed_contour, 1, mode='edge')
                window = np.ones(3) / 3.0
                smoothed_contour = np.convolve(padded_contour, window, mode='valid')
            
            speed_predictions = self.speed_infer.inference(pil_speed)
            
            t_inf = time.time()
            
            # 3. Create clean visualization for front-view with RED contour
            frame_bgr_360_view = frame_bgr_360.copy()
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
                # Setup Top-Down projection parameters
                resolution = 20 # px/m
                W_td = int(40 * resolution)  # 40m wide
                H_td_draw = int(100 * resolution) # 100m vertical drawing height
                H_td_crop = int(50 * resolution)  # 50m vertical display height (closer 50m)
                
                # Load or use default camera matrices
                K_1920 = np.array([[960.0, 0, 960.0], [0, 960.0, 540.0], [0, 0, 1.0]])
                Rt = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 3.25], [1.0, 0.0, -3.2]])
                H_world_to_img_1920 = K_1920 @ Rt
                scale_m = np.array([[640/1920.0, 0, 0], [0, 360/1080.0, 0], [0, 0, 1]])
                H_world_to_img = scale_m @ H_world_to_img_1920
                
                # Matrix: BEV-Pixel -> World-Coordinates (X from 5 to 105)
                M_td2w = np.array([[0, -1.0/resolution, 105.0], [1.0/resolution, 0, -20.0], [0, 0, 1]])
                M_td2img = H_world_to_img @ M_td2w
                
                # clean BEV warping
                topdown_clean = cv2.warpPerspective(frame_bgr_360, M_td2img, (W_td, H_td_draw), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
                
                # 5. Point-based BEV contour mask
                angles = np.linspace(np.pi, 0, len(smoothed_contour))
                img_pts = []
                for idx, angle in zip(smoothed_contour, angles):
                    dist = idx * 10
                    r = (319 - dist * np.sin(angle)) * (360.0/320.0)
                    c = 320 + dist * np.cos(angle)
                    img_pts.append([c, r, 1.0])
                
                M_img2td = np.linalg.inv(M_td2img)
                topdown_mask = np.zeros((H_td_draw, W_td), dtype=np.uint8)
                bev_pts_vis = []
                bev_pts_nav = [] # For filling mask
                for p_img in img_pts:
                    p_td = M_img2td @ p_img
                    u, v = p_td[0]/p_td[2], p_td[1]/p_td[2]
                    
                    v_nav = v
                    if v_nav > H_td_draw: v_nav = float(H_td_draw)
                    bev_pts_nav.append((int(u), int(v_nav)))
                    
                    v_vis = v
                    if v_vis > H_td_draw: v_vis = 0.0 # User requested snap to top
                    bev_pts_vis.append((int(u), int(v_vis)))
                
                # Draw red contour on mask using VIS points
                for i in range(len(bev_pts_vis)-1):
                    p1, p2 = bev_pts_vis[i], bev_pts_vis[i+1]
                    if (-100 < p1[0] < W_td + 100) and (-100 < p1[1] < H_td_draw + 100):
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
                
                # Replace Python logic with C++ processing
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
                    
                    gap_idx1, gap_idx2 = idx1.value, idx2.value
                    self.get_logger().info(f"C++ Target Gap: idx1={gap_idx1}, idx2={gap_idx2}, Total Gaps={num_gaps.value}")

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
                        
                        # Draw on BEV view
                        cv2.line(topdown_clean, (px1, py1), (px2, py2), color, thick)
                        
                        # Draw on front view by projecting (px, py) to front view
                        p1_td = np.array([px1, py1, 1.0])
                        p1_img = M_td2img @ p1_td
                        c_f1, r_f1 = int(p1_img[0]/p1_img[2]), int(p1_img[1]/p1_img[2])
                        
                        p2_td = np.array([px2, py2, 1.0])
                        p2_img = M_td2img @ p2_td
                        c_f2, r_f2 = int(p2_img[0]/p2_img[2]), int(p2_img[1]/p2_img[2])
                        
                        if 0 <= r_f1 < 360 and 0 <= c_f1 < 640 and 0 <= r_f2 < 360 and 0 <= c_f2 < 640:
                            cv2.line(frame_bgr_360_view, (c_f1, r_f1), (c_f2, r_f2), color, thick)
                            
                        if is_target:
                            # Generate a cubic Bezier spline toward the gap center
                            center_px = int((px1 + px2) / 2)
                            center_py = int((py1 + py2) / 2)
                            
                            ego_px, ego_py = W_td // 2, H_td_draw
                            D = ego_py - center_py
                            
                            # Compute control commands
                            X_fwd = (ego_py - center_py) / 20.0 # 20 px/m resolution
                            Y_left = (ego_px - center_px) / 20.0
                            cte = Y_left
                            yaw_error = np.arctan2(Y_left, max(X_fwd, 0.1))
                            
                            # In lead truck mode, match the lead truck's speed; otherwise use fixed target
                            target_speed = self.lead_truck_speed if self.lead_truck_mode else 8.33

                            steer_angle = self.lib_steer.steering_compute(self.steer_handle, cte, yaw_error, target_speed, 0.0)
                            speed_effort = self.lib_pi.pi_compute_effort(self.pi_handle, float(self.current_speed), target_speed)

                            # Clamp the effort to max acceleration limits to avoid infinite windup overriding Ackermann speed target
                            speed_effort = max(min(speed_effort, 1.5), -3.0)

                            msg = AckermannDrive()
                            msg.steering_angle = -float(steer_angle)
                            msg.speed = target_speed
                            msg.acceleration = float(speed_effort)
                            self.ackermann_pub.publish(msg)
                            
                            p0 = np.array([ego_px, ego_py])
                            p1 = np.array([ego_px, ego_py - D * 0.4])
                            p2 = np.array([center_px, center_py + D * 0.4])
                            p3 = np.array([center_px, center_py])
                            
                            t = np.linspace(0, 1, 20)[:, None]
                            curve = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
                            curve = curve.astype(np.int32)
                            
                            # Draw on BEV view (Yellow)
                            cv2.polylines(topdown_clean, [curve], isClosed=False, color=(0, 255, 255), thickness=3)
                            
                            # Draw on front view
                            curve_td = np.hstack((curve, np.ones((len(curve), 1))))
                            p_img_all = (M_td2img @ curve_td.T).T
                            
                            # Filter points that are actually in front of the camera
                            valid_mask = p_img_all[:, 2] > 1e-3
                            if np.any(valid_mask):
                                curve_img = p_img_all[valid_mask]
                                curve_img = curve_img[:, :2] / curve_img[:, 2:]
                                cv2.polylines(frame_bgr_360_view, [curve_img.astype(np.int32)], isClosed=False, color=(0, 255, 255), thickness=3)

                # Crop and resize
                topdown_cropped = topdown_clean[H_td_draw - H_td_crop:, :]
                
                # Resize and output stack
                td_resized = cv2.resize(topdown_cropped, (int(W_td * (540/H_td_crop)), 540), interpolation=cv2.INTER_LINEAR)
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
                f"Inf: {(t_inf-t_prep)*1000:.1f}ms | Draw: {(t_draw-t_inf)*1000:.1f}ms | Ser: {(t_end-t_draw)*1000:.1f}ms"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in process_frame: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
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
    parser.add_argument('--speed_model', type=str, required=True,
                        help='Path to the auto_speed model (best.pt or its directory)')
    parser.add_argument('--lead_truck', type=lambda x: x.lower() == 'true', default=False,
                        help='Enable lead mining truck following mode (default: false). '
                             'Subscribes to /carla/lead_truck/speed and uses it as the target speed.')

    args = parser.parse_args()

    # Ensure paths exist
    if not os.path.exists(args.contour_model):
        print(f"Error: Contour model path not found: {args.contour_model}")
        return
    if not os.path.exists(args.speed_model):
        print(f"Error: AutoSpeed model path not found: {args.speed_model}")
        return

    rclpy.init()
    node = CARLAMiningDemoNode(args.contour_model, args.speed_model, lead_truck=args.lead_truck)
    
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
            if key == 27: # ESC key to exit
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
