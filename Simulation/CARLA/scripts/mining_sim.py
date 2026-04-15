#!/usr/bin/env python3


#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Allows controlling a vehicle with a keyboard."""

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk
    E            : toggle elevation dataset recording (RGB + GT elevation from LiDAR)
    J            : toggle segmentation dataset recording (RGB + semantic label image)

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import os
import re
import sys
import yaml
import queue
import threading

# Remove Isaac Sim from LD_LIBRARY_PATH to avoid spdlog conflicts with ROS2 Humble
if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = re.sub(r'/?home/autoware/isaacsim6/[^:]*:?', '', os.environ["LD_LIBRARY_PATH"])
os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
os.environ.setdefault("ROS_DOMAIN_ID", "42")

import rclpy
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32
from sensor_msgs.msg import Image as RosImage

g_latest_ackermann = None
g_ros_image_pub = None
g_ros_image_queue = queue.Queue(maxsize=1)  # drop old frames if publisher can't keep up

def _ros_image_publisher_thread():
    """Background thread: serialises and publishes BGR images without blocking CARLA callbacks."""
    while True:
        bgr = g_ros_image_queue.get()
        if bgr is None:
            break
        if g_ros_image_pub is None:
            continue
        ros_img = RosImage()
        ros_img.height = bgr.shape[0]
        ros_img.width  = bgr.shape[1]
        ros_img.encoding = 'bgr8'
        ros_img.is_bigendian = False
        ros_img.step = ros_img.width * 3
        ros_img.data = bgr.tobytes()
        g_ros_image_pub.publish(ros_img)

def ackermann_callback(msg):
    global g_latest_ackermann
    g_latest_ackermann = msg

import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import os
import weakref
import time
import cv2

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_e
    from pygame.locals import K_j
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

OBJECT_TO_COLOR = [
    (255, 255, 255),
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142,  35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0,  60, 100),
    (0,  80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (110, 190, 160),
    (170, 120, 50),
    (55, 90, 80),
    (45, 60, 150),
    (157, 234, 50),
    (81, 0, 81),
    (150, 100, 100),
    (230, 150, 140),
    (180, 165, 180),
]

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3, 4]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, traffic_manager, args, client=None):
        self.world = carla_world
        self.sync = args.sync
        self.client = client
        self.traffic_manager = traffic_manager
        self.actor_role_name = args.rolename
        self.lead_truck_enabled = args.lead_truck
        self.num_trucks = args.num_trucks
        self.npc_trucks = []  # list of spawned NPC mining trucks
        self.num_pedestrians = args.num_pedestrian
        self.npc_pedestrians = []  # list of (walker_actor, controller_actor) tuples
        self._ped_walk_data = []   # list of [speed, direction, next_turn_time] for manual walk
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.dataset_recorder = None
        self.seg_recorder = None
        self._seg_config = args.seg_config
        self._tag_config = args.tag_config
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 1
        # Get a random blueprint.
        #blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        #if not blueprint_list:
        #    raise ValueError("Couldn't find any blueprints with the specified filters")
        #blueprint = random.choice(blueprint_list)
        
        blueprint_lib = self.world.get_blueprint_library()
        # 2. Filter for mining earth mover vehicle blueprint
        vehicle_bp_list = blueprint_lib.filter('vehicle.miningtruck.miningtruck*')
        if not vehicle_bp_list:
            vehicle_bp_list = blueprint_lib.filter('*mining*')
        if not vehicle_bp_list:
            raise RuntimeError("Mining earth mover blueprint not found!")
        blueprint = vehicle_bp_list[0]
        
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE5 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
            
        # Deregister and destroy NPC actors safely.
        # Step 1: swap out lists so _tick_pedestrians() and game-loop code see empty lists.
        trucks_to_destroy = self.npc_trucks
        self.npc_trucks = []
        peds_to_destroy = self.npc_pedestrians
        self.npc_pedestrians = []
        self._ped_walk_data = []

        # Step 2: disable autopilot on trucks so the Traffic Manager deregisters them
        # before we destroy the actors (TM runs on its own C++ thread).
        for truck in trucks_to_destroy:
            try:
                truck.set_autopilot(False, self.traffic_manager.get_port())
            except Exception:
                pass

        # Step 3: tick the server so TM deregistration and any pending apply_control
        # commands are fully processed before the actors are removed.
        if trucks_to_destroy or peds_to_destroy:
            try:
                if self.sync:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
            except Exception:
                pass

        # Step 4: destroy all NPC actors atomically via apply_batch (fire-and-forget).
        # Step 3 already flushed all pending apply_control / TM commands with a tick,
        # so the server will process these destroy commands cleanly on its next tick
        # without racing against in-flight C++ thread work.
        # We deliberately use non-sync apply_batch to avoid blocking the game loop.
        actors_to_destroy = list(trucks_to_destroy)
        for walker, controller in peds_to_destroy:
            if controller is not None:
                try:
                    controller.stop()
                except Exception:
                    pass
                actors_to_destroy.append(controller)
            if walker is not None:
                actors_to_destroy.append(walker)

        if actors_to_destroy and self.client is not None:
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in actors_to_destroy])
            except Exception:
                pass
        else:
            for actor in actors_to_destroy:
                try:
                    actor.destroy()
                except Exception:
                    pass

        # Resolve mining truck blueprint once
        npc_bp_list = blueprint_lib.filter('vehicle.miningtruck.miningtruck*')
        if not npc_bp_list:
            npc_bp_list = blueprint_lib.filter('*mining*')

        if npc_bp_list:
            npc_bp = npc_bp_list[0]
            if npc_bp.has_attribute('terramechanics'):
                npc_bp.set_attribute('terramechanics', 'true')

            # --- Lead truck: spawn directly in front of the ego (--lead_truck true) ---
            if self.lead_truck_enabled:
                npc_bp.set_attribute('role_name', 'lead_truck')
                if npc_bp.has_attribute('color'):
                    color = random.choice(npc_bp.get_attribute('color').recommended_values)
                    npc_bp.set_attribute('color', color)

                player_transform = self.player.get_transform()
                forward_vector = player_transform.get_forward_vector()
                spawn_location = player_transform.location + forward_vector * 15.0
                spawn_location.z += 2.0
                lead_spawn = carla.Transform(spawn_location, player_transform.rotation)

                truck = self.world.try_spawn_actor(npc_bp, lead_spawn)
                if truck is not None:
                    try:
                        truck.set_autopilot(True, self.traffic_manager.get_port())
                        self.modify_vehicle_physics(truck)
                        self.traffic_manager.ignore_lights_percentage(truck, 100.0)
                        self.traffic_manager.auto_lane_change(truck, False)
                    except Exception as e:
                        print(f"Warning: lead truck autopilot setup failed: {e}")
                    self.npc_trucks.append(truck)
                else:
                    print("Failed to spawn lead mining truck.")

            # --- Extra trucks: scatter across random map spawn points (--num_trucks N) ---
            if self.num_trucks > 0:
                all_spawn_points = self.map.get_spawn_points()
                # Exclude the ego's spawn point to avoid overlap
                ego_loc = self.player.get_transform().location
                candidates = [sp for sp in all_spawn_points
                              if sp.location.distance(ego_loc) > 20.0]
                random.shuffle(candidates)

                spawned = 0
                for i, sp in enumerate(candidates):
                    if spawned >= self.num_trucks:
                        break
                    npc_bp.set_attribute('role_name', f'npc_truck_{i}')
                    if npc_bp.has_attribute('color'):
                        color = random.choice(npc_bp.get_attribute('color').recommended_values)
                        npc_bp.set_attribute('color', color)
                    truck = self.world.try_spawn_actor(npc_bp, sp)
                    if truck is not None:
                        try:
                            truck.set_autopilot(True, self.traffic_manager.get_port())
                            self.modify_vehicle_physics(truck)
                            self.traffic_manager.ignore_lights_percentage(truck, 100.0)
                            self.traffic_manager.auto_lane_change(truck, False)
                        except Exception as e:
                            print(f"Warning: NPC truck_{i} autopilot setup failed: {e}")
                        self.npc_trucks.append(truck)
                        spawned += 1

                print(f"Spawned {spawned}/{self.num_trucks} requested NPC mining trucks.")

        # --- Pedestrians: spawn walkers with manual random-walk (--num_pedestrian N) ---
        if self.num_pedestrians > 0:
            walker_bp_list = blueprint_lib.filter('walker.pedestrian.*')

            # Prefer navmesh spawn points; fall back to vehicle spawn points.
            nav_locations = [self.world.get_random_location_from_navigation()
                             for _ in range(self.num_pedestrians * 3)]
            nav_locations = [loc for loc in nav_locations if loc is not None]
            if not nav_locations:
                nav_locations = [sp.location for sp in self.map.get_spawn_points()]
            if not nav_locations:
                print("Warning: no spawn locations available for pedestrians.")

            self._ped_walk_data = []
            spawned_peds = 0
            loc_pool = list(nav_locations)
            random.shuffle(loc_pool)
            pool_iter = iter(loc_pool)
            for _ in range(self.num_pedestrians):
                spawn_loc = next(pool_iter, None)
                if spawn_loc is None:
                    break
                walker_bp = random.choice(walker_bp_list)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                walker = self.world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
                if walker is None:
                    continue
                self.npc_pedestrians.append((walker, None))
                angle = random.uniform(0, 2 * math.pi)
                speed = 1.0 + random.random()
                next_turn = time.time() + random.uniform(3.0, 8.0)
                self._ped_walk_data.append([speed, angle, next_turn])
                spawned_peds += 1
            print(f"Spawned {spawned_peds}/{self.num_pedestrians} requested pedestrians.")

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

        # Recreate dataset recorder (preserves recording state across restarts)
        was_recording = self.dataset_recorder.recording if self.dataset_recorder is not None else False
        if self.dataset_recorder is not None:
            self.dataset_recorder.destroy()
        self.dataset_recorder = DatasetRecorder(self.player, self.hud, self._gamma)
        if was_recording:
            self.dataset_recorder.recording = True

        # Recreate segmentation dataset recorder
        was_seg_recording = self.seg_recorder.recording if self.seg_recorder is not None else False
        if self.seg_recorder is not None:
            self.seg_recorder.destroy()
        self.seg_recorder = SegmentationDatasetRecorder(self.player, self.hud, self._seg_config, self._tag_config)
        if was_seg_recording:
            self.seg_recorder.recording = True
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        self.traffic_manager.update_vehicle_lights(self.player, True)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)
        self._tick_pedestrians()

    def _tick_pedestrians(self):
        now = time.time()
        for i, (walker, _) in enumerate(self.npc_pedestrians):
            if walker is None or not walker.is_alive:
                continue
            data = self._ped_walk_data[i]
            speed, angle, next_turn = data
            if now >= next_turn:
                angle = random.uniform(0, 2 * math.pi)
                speed = 1.0 + random.random()
                data[1] = angle
                data[0] = speed
                data[2] = now + random.uniform(3.0, 8.0)
            control = carla.WalkerControl()
            control.speed = speed
            control.direction = carla.Vector3D(math.cos(angle), math.sin(angle), 0.0)
            try:
                walker.apply_control(control)
            except Exception:
                pass

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
            
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]

        # Stop all sensors first, then sleep so in-flight callbacks can drain
        # before the underlying C++ objects are torn down.  Destroying immediately
        # after stop() (without a gap) races with the sensor thread and causes
        # the libc++abi std::exception crash on restart.
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass
        time.sleep(0.1)
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.destroy()
                except Exception:
                    pass

        # Clear sensor references so they aren't destroyed twice
        self.camera_manager.sensor = None
        self.collision_sensor.sensor = None
        self.lane_invasion_sensor.sensor = None
        self.gnss_sensor.sensor = None
        self.imu_sensor.sensor = None

        # Destroy dataset_recorder before the player — its sensors are attached
        # to the player, so player.destroy() would invalidate their C++ objects
        # and cause sensor.stop() to throw std::exception on the next restart.
        if self.dataset_recorder is not None:
            self.dataset_recorder.destroy()
            self.dataset_recorder = None

        if self.seg_recorder is not None:
            self.seg_recorder.destroy()
            self.seg_recorder = None

        if self.player is not None:
            try:
                self.player.destroy()
            except Exception:
                pass
            self.player = None

        trucks_to_destroy = self.npc_trucks
        self.npc_trucks = []
        peds_to_destroy = self.npc_pedestrians
        self.npc_pedestrians = []
        self._ped_walk_data = []

        actors_to_destroy = list(trucks_to_destroy)
        for walker, controller in peds_to_destroy:
            if controller is not None:
                try:
                    controller.stop()
                except Exception:
                    pass
                actors_to_destroy.append(controller)
            if walker is not None:
                actors_to_destroy.append(walker)

        if actors_to_destroy and self.client is not None:
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(x) for x in actors_to_destroy])
            except Exception:
                pass
        else:
            for actor in actors_to_destroy:
                try:
                    actor.destroy()
                except Exception:
                    pass


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_e:
                    world.dataset_recorder.toggle(world.hud)
                elif event.key == K_j:
                    world.seg_recorder.toggle(world.hud)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_f:
                        # Toggle ackermann controller
                        self._ackermann_enabled = not self._ackermann_enabled
                        world.hud.show_ackermann_info(self._ackermann_enabled)
                        world.hud.notification("Ackermann Controller %s" %
                                               ("Enabled" if self._ackermann_enabled else "Disabled"))
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl()
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    world.player.set_light_state(carla.VehicleLightState(current_lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    global g_latest_ackermann
                    if g_latest_ackermann is not None:
                        self._ackermann_control.steer = g_latest_ackermann.steering_angle
                        self._ackermann_control.speed = g_latest_ackermann.speed
                        self._ackermann_control.steering_angle_velocity = g_latest_ackermann.steering_angle_velocity
                        self._ackermann_control.acceleration = g_latest_ackermann.acceleration
                        self._ackermann_control.jerk = g_latest_ackermann.jerk
                        
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

        self._lights = current_lights

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.1, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.lidar.ray_cast_semantic', None, 'Semantic Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def draw_vehicle_3d_bbox(self, image):

        def build_projection_matrix(w, h, fov, is_behind_camera=False):
            focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
            K = np.identity(3)

            if is_behind_camera:
                K[0, 0] = K[1, 1] = -focal
            else:
                K[0, 0] = K[1, 1] = focal

            K[0, 2] = w / 2.0
            K[1, 2] = h / 2.0
            return K

        def get_image_point(loc, K, w2c):
            # Calculate 2D projection of 3D coordinate

            # Format the input coordinate (loc is a carla.Position object)
            point = np.array([loc.x, loc.y, loc.z, 1])
            # transform to camera coordinates
            point_camera = np.dot(w2c, point)

            # New we must change from UE4's coordinate system to an "standard"
            # (x, y ,z) -> (y, -z, x)
            # and we remove the fourth componebonent also
            point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

            # now project 3D->2D using the camera matrix
            point_img = np.dot(K, point_camera)
            # normalize
            point_img[0] /= point_img[2]
            point_img[1] /= point_img[2]

            return point_img[0:2]
        
        def point_in_canvas(pos, img_h, img_w):
            """Return true if point is in canvas"""
            if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
                return True
            return False

        # Get the world to camera matrix
        world_2_camera = np.array(self.sensor.get_transform().get_inverse_matrix())

        camera_bp = self.sensors[self.index][-1]

        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        # Calculate the camera projection matrix to project from 3D -> 2D
        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        world = self._parent.get_world()
              
        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != self._parent.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(self._parent.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < 80:

                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the other vehicle. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = self._parent.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - self._parent.get_transform().location

                    if forward_vec.dot(ray) > 0:
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        for edge in edges:
                            p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                            p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

                            p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                            if not p1_in_canvas and not p2_in_canvas:
                                continue

                            ray0 = verts[edge[0]] - self.sensor.get_transform().location
                            ray1 = verts[edge[1]] - self.sensor.get_transform().location
                            cam_forward_vec = self.sensor.get_transform().get_forward_vector()

                            # One of the vertex is behind the camera
                            if not (cam_forward_vec.dot(ray0) > 0):
                                p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                            if not (cam_forward_vec.dot(ray1) > 0):
                                p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)

                            cv2.line(bgr_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)        

        #cv2.imshow('3D BBox', bgr_image)
        #cv2.waitKey(10)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0] == 'sensor.lidar.ray_cast':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0] == 'sensor.lidar.ray_cast_semantic':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 6), 6))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            for i in range(len(image)):
                point = lidar_data[i]
                lidar_tag = image[i].object_tag
                lidar_img[tuple(point.T)] = OBJECT_TO_COLOR[int(lidar_tag)]
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_array = np.ascontiguousarray(array[:, :, :3])   # BGRA -> BGR
            array = bgr_array[:, :, ::-1]                        # BGR  -> RGB for pygame
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            # self.draw_vehicle_3d_bbox(array)
            try:
                # Downscale to 640×360 before publishing (0.69 MB vs 6.2 MB).
                # The demo consumes 640×360 anyway, so this avoids 9× extra DDS traffic.
                bgr_360 = cv2.resize(bgr_array, (640, 360), interpolation=cv2.INTER_LINEAR)
                g_ros_image_queue.put_nowait(bgr_360)
            except queue.Full:
                pass  # drop frame — publisher thread hasn't caught up
            
        if self.recording:
            image.save_to_disk(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '_rgb', '%08d' % image.frame))


class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_lidar_image)

            return lidar
        
        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_semanticlidar_image)

            return lidar
        
        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar
        
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = np.ascontiguousarray(array)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_semanticlidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


# ==============================================================================
# -- DatasetRecorder -----------------------------------------------------------
# ==============================================================================


class DatasetRecorder:
    """
    Records paired RGB images and ground-truth elevation maps for terrain
    elevation network training. Toggle with the E key.

    Spawns a dedicated front-facing RGB camera and LiDAR sensor. On each
    frame where both sensors fire at the same simulation frame, saves:
      <output_dir>/rgb/<frame>.png        -- RGB image
      <output_dir>/elevation/<frame>.npy  -- float32 height map (NaN = no data)
    """

    def __init__(self, parent_actor, hud, gamma_correction, output_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '_elevation_dataset')):
        self.parent = parent_actor
        self.output_dir = output_dir
        self.recording = False
        self.frame_count = 0       # set on toggle-on via _next_frame_index()
        self._last_save_time = 0.0 # wall-clock time of last saved pair (2 Hz gate)
        self.IMG_W = hud.dim[0]
        self.IMG_H = hud.dim[1]

        self._rgb_array = None
        self._rgb_frame = -1
        self._depth_array = None   # float32 (H, W) depth in metres
        self._depth_frame = -1
        self._lidar_points = None
        self._lidar_frame = -1

        world = parent_actor.get_world()
        bp_lib = world.get_blueprint_library()

        # Match the default CameraManager position (transform_index=1):
        # x=+0.8*bound_x, z=1.3*bound_z, Rigid attachment
        bound_x = 0.5 + parent_actor.bounding_box.extent.x
        bound_z = 0.5 + parent_actor.bounding_box.extent.z
        sensor_transform = carla.Transform(
            carla.Location(x=+0.8*bound_x, y=0.0, z=1.3*bound_z),
            carla.Rotation(pitch=0.0)
        )

        # Initialise sensor handles to None so destroy() is always safe
        self.camera       = None
        self.depth_camera = None
        self.lidar        = None

        # RGB camera — same resolution, gamma, and FOV as the display camera
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.IMG_W))
        cam_bp.set_attribute('image_size_y', str(self.IMG_H))
        if cam_bp.has_attribute('gamma'):
            cam_bp.set_attribute('gamma', str(gamma_correction))
        self.fov = cam_bp.get_attribute('fov').as_float()
        try:
            self.camera = world.spawn_actor(cam_bp, sensor_transform, attach_to=parent_actor)
        except Exception as e:
            print(f'DatasetRecorder: failed to spawn RGB camera: {e}')

        # Pre-compute the 3×4 projection matrix P = K @ [R|t] at net resolution (640×320).
        # CARLA LiDAR frame: x=fwd, y=left, z=up.
        # OpenCV camera frame: x=right, y=down, z=fwd → cam_x=-lidar_y, cam_y=-lidar_z, cam_z=lidar_x
        # Rotation R (LiDAR → camera):
        #   R = [[ 0, -1,  0],
        #        [ 0,  0, -1],
        #        [ 1,  0,  0]]
        # Camera and LiDAR are co-mounted → t = [0, 0, 0]
        # Intrinsics at net resolution (640×320):
        #   f_net = NET_W / (2 * tan(fov/2))
        #   cx_net = NET_W / 2,  cy_net = NET_H / 2
        # P = K @ [R|t] (flattened row-major to shape (12,))
        f_net  = self._NET_W / (2.0 * np.tan(np.radians(self.fov / 2.0)))
        cx_net = self._NET_W / 2.0
        cy_net = self._NET_H / 2.0
        K = np.array([[f_net, 0.0,   cx_net],
                      [0.0,   f_net, cy_net],
                      [0.0,   0.0,   1.0   ]], dtype=np.float32)
        Rt = np.array([[ 0.0, -1.0,  0.0, 0.0],
                       [ 0.0,  0.0, -1.0, 0.0],
                       [ 1.0,  0.0,  0.0, 0.0]], dtype=np.float32)
        self._cam_params = (K @ Rt).flatten()  # shape (12,)

        # Precompute per-pixel (v-cy) and (u-cx) grids at net resolution.
        # Used to back-project depth pixels → 3-D world points.
        v_net, u_net = np.mgrid[0:self._NET_H, 0:self._NET_W]
        self._v_minus_cy = (v_net - cy_net).astype(np.float32)  # (NET_H, NET_W)
        self._u_minus_cx = (u_net - cx_net).astype(np.float32)  # (NET_H, NET_W)

        # Depth camera — same mount, resolution, and FOV as the RGB camera
        # CARLA depth encoding: depth_m = (R + G*256 + B*65536) / (256^3 - 1) * 1000
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.IMG_W))
        depth_bp.set_attribute('image_size_y', str(self.IMG_H))
        depth_bp.set_attribute('fov', str(self.fov))
        try:
            self.depth_camera = world.spawn_actor(depth_bp, sensor_transform, attach_to=parent_actor)
        except Exception as e:
            print(f'DatasetRecorder: failed to spawn depth camera: {e}')

        # LiDAR — only set range; leave other attributes at blueprint defaults
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        try:
            self.lidar = world.spawn_actor(lidar_bp, sensor_transform, attach_to=parent_actor)
        except Exception as e:
            print(f'DatasetRecorder: failed to spawn LiDAR: {e}')

        weak_self = weakref.ref(self)
        if self.camera is not None:
            self.camera.listen(lambda img: DatasetRecorder._on_rgb(weak_self, img))
        if self.depth_camera is not None:
            self.depth_camera.listen(lambda img: DatasetRecorder._on_depth(weak_self, img))
        if self.lidar is not None:
            self.lidar.listen(lambda pts: DatasetRecorder._on_lidar(weak_self, pts))

    def _next_frame_index(self):
        """Scan images/ and return the next available frame index (0 if empty)."""
        max_idx = -1
        img_dir = os.path.join(self.output_dir, 'images')
        if os.path.isdir(img_dir):
            for fname in os.listdir(img_dir):
                stem, _ = os.path.splitext(fname)
                try:
                    idx = int(stem)
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    pass
        return max_idx + 1

    def toggle(self, hud=None):
        self.recording = not self.recording
        if self.recording:
            os.makedirs(os.path.join(self.output_dir, 'images'),              exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'gt_lidar_elevations'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'gt_depth_elevations'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'gt_bev_lidar_elevations'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'gt_bev_depth_elevations'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'camera_params'),       exist_ok=True)
            # Save shared camera params once (same for every frame)
            cam_path = os.path.join(self.output_dir, 'camera_params', 'camera_params.npy')
            if not os.path.exists(cam_path):
                np.save(cam_path, self._cam_params)
            self.frame_count = self._next_frame_index()
            self._last_save_time = 0.0  # let first frame save immediately
            msg = 'Elevation Dataset Recording ON -> %s (next frame: %06d)' % (
                self.output_dir, self.frame_count)
        else:
            msg = 'Elevation Dataset Recording OFF (%d frames saved)' % self.frame_count
        print(msg)
        if hud:
            hud.notification(msg)

    @staticmethod
    def _on_rgb(weak_self, image):
        self = weak_self()
        if not self:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        self._rgb_array = arr.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA -> BGR
        self._rgb_frame = image.frame

    @staticmethod
    def _on_depth(weak_self, image):
        self = weak_self()
        if not self:
            return
        # CARLA depth raw format: BGRA where depth_m = (R + G*256 + B*65536)/(256^3-1)*1000
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        R = arr[:, :, 2].astype(np.float32)
        G = arr[:, :, 1].astype(np.float32)
        B = arr[:, :, 0].astype(np.float32)
        self._depth_array = (R + G * 256.0 + B * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0
        self._depth_frame = image.frame

    @staticmethod
    def _on_lidar(weak_self, point_cloud):
        self = weak_self()
        if not self:
            return
        pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape((-1, 4))
        self._lidar_points = pts
        self._lidar_frame = point_cloud.frame
        # Save when LiDAR fires (slower sensor) if a recent RGB frame is available.
        # Allow up to 2-frame gap to handle async mode where frame numbers differ.
        if (self.recording
                and self._rgb_array is not None
                and abs(self._rgb_frame - self._lidar_frame) <= 2):
            self._save_pair()

    _NET_W  = 640
    _NET_H  = 320
    # Elevation bin constants — must match LoadDataElevation / elevation_head.py
    _H_MIN   = -0.50  # metres
    _H_STEP  =  0.05  # metres
    _NUM_BINS = 40
    # BEV grid parameters
    _BEV_RES    = 10    # pixels per metre
    _BEV_X_MAX  = 60.0  # forward range (metres)
    _BEV_Y_HALF = 30.0  # half lateral range (metres); total width = 60 m

    def _elev_to_png(self, elev_map):
        """Convert a float32 elevation map (metres, NaN=empty) to a scaled uint8 PNG.
        Valid pixels: binned 0–39 then scaled to 96–255. Empty pixels: 0."""
        valid = ~np.isnan(elev_map)
        filled = np.where(valid, elev_map, self._H_MIN)
        bins = np.floor((filled - self._H_MIN) / self._H_STEP).astype(np.int32)
        bins = np.clip(bins, 0, self._NUM_BINS - 1)
        out = (bins * 159 // (self._NUM_BINS - 1) + 96).astype(np.uint8)
        out[~valid] = 0
        return out

    @staticmethod
    def _fill_holes(elev_map):
        """Fill NaN holes in a float32 elevation map using nearest-valid-neighbour.

        Uses a distance transform to find the closest valid pixel for each empty
        pixel.  Works in float32 before bin-encoding so no precision is lost.
        If the map is entirely empty, it is returned unchanged (all-NaN)."""
        from scipy.ndimage import distance_transform_edt
        valid = ~np.isnan(elev_map)
        if not valid.any():
            return elev_map
        _, idx = distance_transform_edt(~valid, return_indices=True)
        return elev_map[idx[0], idx[1]]

    def _bev_from_lidar(self, lidar_points):
        """Splat LiDAR (x=fwd, y=left, z=up) into a BEV elevation grid.
        Returns float32 (BEV_H, BEV_W) with NaN where no points land."""
        BEV_H = int(self._BEV_X_MAX * self._BEV_RES)
        BEV_W = int(self._BEV_Y_HALF * 2 * self._BEV_RES)

        x = lidar_points[:, 0]  # fwd
        y = lidar_points[:, 1]  # left
        z = lidar_points[:, 2]  # up = elevation

        # BEV row: fwd distance, near at bottom → row decreases with x
        row = (BEV_H - 1 - (x * self._BEV_RES)).astype(np.int32)
        # BEV col: left is positive → col decreases with y
        col = (self._BEV_Y_HALF * self._BEV_RES - y * self._BEV_RES).astype(np.int32)

        valid = (x > 0.5) & (row >= 0) & (row < BEV_H) & (col >= 0) & (col < BEV_W)
        row, col, z_v, x_v = row[valid], col[valid], z[valid], x[valid]

        # Far-to-near: closer points overwrite farther ones
        order = np.argsort(x_v)
        bev = np.full((BEV_H, BEV_W), np.nan, dtype=np.float32)
        bev[row[order], col[order]] = z_v[order]
        return bev

    def _bev_from_depth(self, depth_net):
        """Back-project depth image → BEV elevation grid using camera_params.
        P = cam_params.reshape(3,4): extracts f, cx, cy.
        World frame: x=fwd (depth), y=left, z=up (elevation).
        Returns float32 (BEV_H, BEV_W) with NaN where depth > 50 m."""
        BEV_H = int(self._BEV_X_MAX * self._BEV_RES)
        BEV_W = int(self._BEV_Y_HALF * 2 * self._BEV_RES)

        # Extract intrinsics from camera_params: P=[[cx,-f,0,0],[cy,0,-f,0],[1,0,0,0]]
        P = self._cam_params.reshape(3, 4)
        f  = -P[0, 1]   # focal length at net resolution
        cx =  P[0, 0]
        cy =  P[1, 0]

        d = depth_net  # (NET_H, NET_W), capped at 50 m
        world_x = d                                              # forward
        world_y = -(self._u_minus_cx * d / f)                   # left
        world_z = -(self._v_minus_cy * d / f)                   # up = elevation

        row = (BEV_H - 1 - (world_x * self._BEV_RES)).astype(np.int32)
        col = (self._BEV_Y_HALF * self._BEV_RES - world_y * self._BEV_RES).astype(np.int32)

        row, col = row.ravel(), col.ravel()
        world_x_r, world_z_r = world_x.ravel(), world_z.ravel()

        valid = (world_x_r > 0.5) & (row >= 0) & (row < BEV_H) & (col >= 0) & (col < BEV_W)
        row, col = row[valid], col[valid]
        x_v, z_v = world_x_r[valid], world_z_r[valid]

        order = np.argsort(x_v)
        bev = np.full((BEV_H, BEV_W), np.nan, dtype=np.float32)
        bev[row[order], col[order]] = z_v[order]
        return bev

    def _save_pair(self):
        # 2 Hz gate: skip if less than 0.5 s since the last save
        now = time.time()
        if now - self._last_save_time < 0.5:
            return
        self._last_save_time = now

        elevation = self._build_elevation_map()

        # --- images/<frame>.png  (640×320 BGR) ---
        rgb_resized = cv2.resize(self._rgb_array,
                                 (self._NET_W, self._NET_H),
                                 interpolation=cv2.INTER_LINEAR)

        # --- gt_lidar_elevations ---
        elev_resized = cv2.resize(elevation,
                                  (self._NET_W, self._NET_H),
                                  interpolation=cv2.INTER_NEAREST)
        elev_valid = ~np.isnan(elev_resized)   # which pixels had a direct LiDAR hit
        elev_resized = self._fill_holes(elev_resized)
        gt_lidar = self._elev_to_png(elev_resized)
        gt_lidar[~elev_valid] = 0              # sky / no-hit pixels must stay 0

        # --- gt_depth_elevations ---
        # Back-project: elev = -(v - cy) * depth / f.  Cap at 50 m (matches LiDAR).
        if self._depth_array is not None:
            depth_net = cv2.resize(self._depth_array,
                                   (self._NET_W, self._NET_H),
                                   interpolation=cv2.INTER_NEAREST)
            depth_net = np.clip(depth_net, 0.0, 60.0)
            P = self._cam_params.reshape(3, 4)
            f_net = -P[0, 1]
            depth_elev = -(self._v_minus_cy * depth_net / f_net).astype(np.float32)
            gt_depth = self._elev_to_png(depth_elev)
        else:
            depth_net = None
            gt_depth  = np.zeros((self._NET_H, self._NET_W), dtype=np.uint8)

        # --- gt_bev_lidar_elevations ---
        bev_lidar    = self._bev_from_lidar(self._lidar_points)
        gt_bev_lidar = self._elev_to_png(bev_lidar)

        # --- gt_bev_depth_elevations ---
        BEV_H = int(self._BEV_X_MAX * self._BEV_RES)
        BEV_W = int(self._BEV_Y_HALF * 2 * self._BEV_RES)
        if depth_net is not None:
            bev_depth    = self._bev_from_depth(depth_net)
            gt_bev_depth = self._elev_to_png(bev_depth)
        else:
            gt_bev_depth = np.zeros((BEV_H, BEV_W), dtype=np.uint8)

        # --- camera_params/ (written once in toggle) ---

        stem = '%06d' % self.frame_count
        cv2.imwrite(os.path.join(self.output_dir, 'images',              stem + '.png'), rgb_resized)
        cv2.imwrite(os.path.join(self.output_dir, 'gt_lidar_elevations', stem + '.png'), cv2.flip(gt_lidar, 1))
        cv2.imwrite(os.path.join(self.output_dir, 'gt_depth_elevations', stem + '.png'), gt_depth)
        cv2.imwrite(os.path.join(self.output_dir, 'gt_bev_lidar_elevations', stem + '.png'), cv2.flip(gt_bev_lidar, 1))
        cv2.imwrite(os.path.join(self.output_dir, 'gt_bev_depth_elevations', stem + '.png'), gt_bev_depth)
        self.frame_count += 1

    def _build_elevation_map(self):
        """
        Project LiDAR points (sensor frame: x=fwd, y=left, z=up) into the
        camera image plane and fill a float32 height map with z values.
        Camera and LiDAR share the same mount transform so no extrinsic
        offset is needed between them.
        """
        focal = self.IMG_W / (2.0 * np.tan(np.radians(self.fov / 2.0)))
        cx, cy = self.IMG_W / 2.0, self.IMG_H / 2.0

        pts = self._lidar_points  # (N, 4): x fwd, y left, z up, intensity
        depth = pts[:, 0]         # forward distance = depth in camera z-axis

        # Keep only points in front of sensor
        mask = depth > 0.5
        if not np.any(mask):
            return np.full((self.IMG_H, self.IMG_W), np.nan, dtype=np.float32)

        x = pts[mask, 0]  # fwd
        y = pts[mask, 1]  # left
        z = pts[mask, 2]  # up
        d = depth[mask]

        # Project to image plane
        # CARLA camera: u increases rightward (-y direction), v increases downward (-z direction)
        u = (-y * focal / d + cx).astype(np.int32)
        v = (-z * focal / d + cy).astype(np.int32)

        in_frame = (u >= 0) & (u < self.IMG_W) & (v >= 0) & (v < self.IMG_H)
        u, v, z, d = u[in_frame], v[in_frame], z[in_frame], d[in_frame]

        elevation = np.full((self.IMG_H, self.IMG_W), np.nan, dtype=np.float32)

        # Splat far-to-near so nearest point wins per pixel
        order = np.argsort(d)[::-1]
        elevation[v[order], u[order]] = z[order]

        return elevation

    def destroy(self):
        # Stop recording first so in-flight callbacks don't call _save_pair
        # after sensors have been destroyed (prevents std::exception on restart)
        self.recording = False
        # Stop all sensors before destroying any — lets in-flight callbacks drain
        # before the underlying C++ objects are torn down (avoids std::exception
        # crash when Backspace is pressed while a LiDAR/depth callback is live).
        for sensor in (self.camera, self.depth_camera, self.lidar):
            if sensor is not None:
                sensor.stop()
        time.sleep(0.1)
        for sensor in (self.camera, self.depth_camera, self.lidar):
            if sensor is not None:
                try:
                    sensor.destroy()
                except Exception:
                    pass
        self.camera = None
        self.depth_camera = None
        self.lidar = None


# ==============================================================================
# -- SegmentationDatasetRecorder -----------------------------------------------
# ==============================================================================


class SegmentationDatasetRecorder:
    """
    Records paired RGB images and semantic segmentation label images.
    Toggle with the J key.

    Spawns a dedicated front-facing RGB camera and a semantic segmentation
    camera at the same mount point.  On each frame where both fire together,
    saves:
      <output_dir>/images/<frame>.png  -- source RGB image
      <output_dir>/labels/<frame>.png  -- RGB-encoded semantic label image

    Tag mapping (CARLA raw tag → objectseg class ID) is loaded from
    tag_config_path (carla_objectseg.yaml by default).
    Color mapping (class ID → RGB) is loaded from seg_config_path
    (objectseg.yaml by default).  Any unmapped tag defaults to class 0
    (background).
    """

    def __init__(self, parent_actor, hud, seg_config_path, tag_config_path,
                 output_dir=os.path.join(
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'data', '_segmentation_dataset')):
        self.parent = parent_actor
        self.output_dir = output_dir
        self.recording = False
        self.frame_count = 0
        self._last_save_time = 0.0
        self.IMG_W = 1280
        self.IMG_H = 720

        self._rgb_array = None
        self._rgb_frame = -1
        self._seg_array = None   # uint8 (H, W) — class ID per pixel
        self._seg_frame = -1

        # Load tag mapping from YAML: raw CARLA tag -> objectseg class ID
        self._tag_lut = SegmentationDatasetRecorder._load_tag_lut(tag_config_path)
        # Load color map from YAML: class_id -> BGR
        self._color_lut = SegmentationDatasetRecorder._load_color_lut(seg_config_path)

        # Resolve dst dirs from tag config (paths relative to repo root)
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            with open(tag_config_path, 'r') as _f:
                _cfg = yaml.safe_load(_f)
            self.image_dir = os.path.join(repo_root, _cfg['dst_image_dir'])
            self.gt_mask_dir = os.path.join(repo_root, _cfg['dst_gt_mask_dir'])
        except Exception as e:
            print(f'SegmentationDatasetRecorder: dst dirs not found in tag config, using defaults: {e}')
            self.image_dir = os.path.join(output_dir, 'images')
            self.gt_mask_dir = os.path.join(output_dir, 'gt_masks')

        world = parent_actor.get_world()
        bp_lib = world.get_blueprint_library()

        bound_x = 0.5 + parent_actor.bounding_box.extent.x
        bound_z = 0.5 + parent_actor.bounding_box.extent.z
        sensor_transform = carla.Transform(
            carla.Location(x=+0.8 * bound_x, y=0.0, z=1.3 * bound_z),
            carla.Rotation(pitch=0.0)
        )

        self.camera = None
        self.seg_camera = None

        # RGB camera
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.IMG_W))
        cam_bp.set_attribute('image_size_y', str(self.IMG_H))
        try:
            self.camera = world.spawn_actor(cam_bp, sensor_transform, attach_to=parent_actor)
        except Exception as e:
            print(f'SegmentationDatasetRecorder: failed to spawn RGB camera: {e}')

        # Semantic segmentation camera (raw — red channel = semantic tag ID)
        seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(self.IMG_W))
        seg_bp.set_attribute('image_size_y', str(self.IMG_H))
        try:
            self.seg_camera = world.spawn_actor(seg_bp, sensor_transform, attach_to=parent_actor)
        except Exception as e:
            print(f'SegmentationDatasetRecorder: failed to spawn seg camera: {e}')

        weak_self = weakref.ref(self)
        if self.camera is not None:
            self.camera.listen(lambda img: SegmentationDatasetRecorder._on_rgb(weak_self, img))
        if self.seg_camera is not None:
            self.seg_camera.listen(lambda img: SegmentationDatasetRecorder._on_seg(weak_self, img))

    @staticmethod
    def _load_color_lut(config_path):
        """Load YAML color map and return a (256, 3) uint8 LUT (class_id -> BGR)."""
        lut = np.zeros((256, 3), dtype=np.uint8)
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            mapping = cfg.get('id_to_color_mapping', {})
            for class_id, rgb in mapping.items():
                if 0 <= class_id < 256:
                    # Config stores RGB; OpenCV uses BGR
                    lut[class_id] = [rgb[2], rgb[1], rgb[0]]
        except Exception as e:
            print(f'SegmentationDatasetRecorder: failed to load color map "{config_path}": {e}')
        return lut

    @staticmethod
    def _load_tag_lut(config_path):
        """Load YAML tag mapping and return a 256-entry uint8 LUT (CARLA tag -> class ID)."""
        lut = np.zeros(256, dtype=np.uint8)
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            mapping = cfg.get('carla_tag_to_class', {})
            for tag, class_id in mapping.items():
                if 0 <= tag < 256:
                    lut[tag] = int(class_id)
        except Exception as e:
            print(f'SegmentationDatasetRecorder: failed to load tag map "{config_path}": {e}')
        return lut

    def _next_frame_index(self):
        max_idx = -1
        if os.path.isdir(self.image_dir):
            for fname in os.listdir(self.image_dir):
                stem, _ = os.path.splitext(fname)
                try:
                    idx = int(stem)
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    pass
        return max_idx + 1

    def toggle(self, hud=None):
        self.recording = not self.recording
        if self.recording:
            os.makedirs(self.image_dir, exist_ok=True)
            os.makedirs(self.gt_mask_dir, exist_ok=True)
            self.frame_count = self._next_frame_index()
            self._last_save_time = 0.0
            msg = 'Segmentation Dataset Recording ON -> %s (next frame: %06d)' % (
                self.image_dir, self.frame_count)
        else:
            msg = 'Segmentation Dataset Recording OFF (%d frames saved)' % self.frame_count
        print(msg)
        if hud:
            hud.notification(msg)

    @staticmethod
    def _on_rgb(weak_self, image):
        self = weak_self()
        if not self:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        self._rgb_array = arr.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA -> BGR
        self._rgb_frame = image.frame

    @staticmethod
    def _on_seg(weak_self, image):
        self = weak_self()
        if not self:
            return
        # Raw semantic segmentation: BGRA where red channel = semantic tag ID
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        tag_ids = arr[:, :, 2]                          # red channel: CARLA tag ID
        self._seg_array = self._tag_lut[tag_ids]        # map tag -> class ID
        self._seg_frame = image.frame
        if (self.recording
                and self._rgb_array is not None
                and abs(self._rgb_frame - self._seg_frame) <= 2):
            self._save_pair()

    def _save_pair(self):
        now = time.time()
        if now - self._last_save_time < 0.5:
            return
        self._last_save_time = now

        stem = '%06d' % self.frame_count

        # Source RGB image
        cv2.imwrite(os.path.join(self.image_dir, stem + '.png'), self._rgb_array)

        # Label image: map class IDs -> BGR colors via LUT
        label_bgr = self._color_lut[self._seg_array]   # (H, W, 3) BGR
        cv2.imwrite(os.path.join(self.gt_mask_dir, stem + '.png'), label_bgr)

        self.frame_count += 1

    def destroy(self):
        self.recording = False
        for sensor in (self.camera, self.seg_camera):
            if sensor is not None:
                sensor.stop()
        time.sleep(0.1)
        for sensor in (self.camera, self.seg_camera):
            if sensor is not None:
                try:
                    sensor.destroy()
                except Exception:
                    pass
        self.camera = None
        self.seg_camera = None


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    
    # Initialize ROS 2
    rclpy.init(args=None)
    ros_node = rclpy.create_node('mining_sim_bridge')
    ros_node.create_subscription(AckermannDrive, '/carla/ego_vehicle/ackermann_cmd', ackermann_callback, 10)
    speed_pub = ros_node.create_publisher(Float32, '/carla/ego_vehicle/speed', 10)
    global g_ros_image_pub
    g_ros_image_pub = ros_node.create_publisher(RosImage, '/carla/ego_vehicle/rgb/image', 10)
    img_pub_thread = threading.Thread(target=_ros_image_publisher_thread, daemon=True)
    img_pub_thread.start()

    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        world = client.load_world('Mine_01')
        sim_world = client.get_world()
        traffic_manager = client.get_trafficmanager()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, traffic_manager, args, client)
        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            rclpy.spin_once(ros_node, timeout_sec=0.0)
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            
            # Publish current ego vehicle speed if available
            if world.player is not None:
                v = world.player.get_velocity()
                speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
                msg = Float32()
                msg.data = speed
                speed_pub.publish(msg)
                
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot', action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1920x1080',
        help='window resolution (default: 1920x1080)')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All',
        help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '--rolename', metavar='NAME', default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma', default=1.0, type=float,
        help='Gamma correction of the camera (default: 1.0)')
    argparser.add_argument(
        '--sync', action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--lead_truck', type=lambda x: x.lower() == 'true', default=False,
        help='Spawn a lead mining truck in front of the ego vehicle (default: false)')
    argparser.add_argument(
        '--num_trucks', type=int, default=0, metavar='N',
        help='Number of additional mining trucks to spawn at random map locations with autopilot (default: 0)')
    argparser.add_argument(
        '--num_pedestrian', type=int, default=0, metavar='N',
        help='Number of pedestrians to spawn with AI controllers (default: 0)')
    argparser.add_argument(
        '--seg_config', metavar='PATH',
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'Models', 'data_parsing', 'config', 'color_map', 'objectseg.yaml'),
        help='Path to segmentation color map YAML config (default: Models/data_parsing/config/color_map/objectseg.yaml)')
    argparser.add_argument(
        '--tag_config', metavar='PATH',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'config', 'carla_objectseg.yaml'),
        help='Path to CARLA tag→class mapping YAML (default: scripts/config/carla_objectseg.yaml)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()