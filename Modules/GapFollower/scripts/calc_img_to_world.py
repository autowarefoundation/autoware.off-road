import numpy as np
import yaml
import os
import argparse

def compute_homography(K, R, t):
    '''
    Compute the homography matrix for the ground plane (z=0).
    K: Camera Intrinsic Matrix (3x3)
    R: Camera Rotation Matrix (3x3)
    t: Camera Translation Vector (3x1)
    '''
    # The homography matrix maps ground plane points (X, Y, 1) to image plane points (u, v, 1) scaling.
    # H = K * [r1, r2, t] where r1, r2 are the first two columns of R
    r1 = R[:, 0:1]
    r2 = R[:, 1:2]
    # concatenate r1, r2, t
    Rt = np.hstack((r1, r2, t))
    H = np.dot(K, Rt)
    
    # Normalize H
    H = H / H[2, 2]
    return H

def euler_to_rot_matrix(pitch, yaw, roll):
    '''
    Convert Euler angles (in radians) to rotation matrix.
    Assumes rotation order: yaw (Z), pitch (Y), roll (X).
    '''
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz.dot(Ry).dot(Rx)

def main():
    parser = argparse.ArgumentParser(description='Calculate Image to World (Ground Plane) Homography Matrix.')
    parser.add_argument('--output', type=str, default='../test/image_to_world_transform.yaml', help='Output YAML file path')
    args = parser.parse_args()

    # Dummy Intrinsic camera parameters
    # For CARLA mining vehicle (1920x1080 resolution, 90 deg FOV)
    # fx = fy = w / (2.0 * tan(fov * pi / 360.0)) = 1920 / (2.0 * tan(90 * pi / 360.0)) = 960.0
    fx, fy = 960.0, 960.0
    cx, cy = 960.0, 540.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # Dummy Extrinsic parameters
    # For CARLA mining vehicle rigid camera (index 1), pitch = 0.
    # Approx bounding box extents for an earth mover: x=3.5m, z=2.0m -> bound_x=4.0m, bound_z=2.5m
    # Camera position per mining_sim.py: x = +0.8 * bound_x = 3.2m, z = +1.3 * bound_z = 3.25m
    x_offset = 3.2
    height = 3.25
    pitch = 0.0
    yaw = 0.0
    roll = 0.0

    # Correct mapping for forward-looking camera:
    # World: X forward, Y right, Z up. Ground is Z=0.
    # Camera (CV convention): Z forward, X right, Y down.
    # Camera position in World: [x_offset, 0, height]
    # P_cam = R_w2c * (P_world - P_cam_in_world)
    # R_w2c should map:
    # World +X -> Camera +Z
    # World +Y -> Camera +X
    # World -Z -> Camera +Y
    R_w2c = np.array([
        [0, 1, 0],   # Cam X is World Y
        [0, 0, -1],  # Cam Y is World -Z
        [1, 0, 0]    # Cam Z is World X
    ])
    
    # Standard Euler rotation from camera base to tilted camera
    R_tilt = euler_to_rot_matrix(pitch, yaw, roll)
    R = R_tilt @ R_w2c
    
    T_cam_in_world = np.array([[x_offset], [0], [height]])
    t = -R @ T_cam_in_world

    # H = K * [r1, r2, t] maps [X_world, Y_world, 1] (at Z=0) to [u, v, scale]
    H_img_to_world_proj = compute_homography(K, R, t)
    
    # The matrix H maps (X_world, Y_world, 1) -> (u, v, scale)
    # We want IPM matrix that goes from Image Pixel -> World Coordinate
    # So we want the inverse of H
    try:
        ipm_matrix = np.linalg.inv(H_img_to_world_proj)
        ipm_matrix = ipm_matrix / ipm_matrix[2, 2] # Normalize
    except np.linalg.LinAlgError:
        print("Error: Homography matrix is singular.")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving IPM matrix to {args.output}")
    with open(args.output, 'w') as f:
        yaml.dump({
            "camera_matrix": K.tolist(),
            "rotation_matrix": R.tolist(),
            "translation_vector": t.tolist(),
            "homography_world_to_img": H_img_to_world_proj.tolist(),
            "ipm_img_to_world": ipm_matrix.tolist()
        }, f, default_flow_style=False)

if __name__ == '__main__':
    main()
