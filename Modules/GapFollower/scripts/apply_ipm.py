import cv2
import yaml
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='../test/test.png')
    parser.add_argument('--yaml', default='../test/image_to_world_transform.yaml')
    parser.add_argument('--output', default='../test/topdown_test.png')
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image: {args.image}")
        return

    h_img_src, w_img_src = img.shape[:2]
    # The homography in YAML assumes 1920x1080. If the image is different, we must scale K.
    # test.png might be 1280x720, so let's scale the source points.
    scale_x = w_img_src / 1920.0
    scale_y = h_img_src / 1080.0
    
    # M_scale maps 1920 image to actual image
    M_scale = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])

    # Load yaml
    with open(args.yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # M_world_to_img_1920
    H_world_to_img_1920 = np.array(data['homography_world_to_img'])
    
    # M_world_to_img_actual
    H_world_to_img = M_scale @ H_world_to_img_1920
    
    # The H matrix maps (X_world, Y_world, 1) -> (u, v, scale)
    # The debug logs show:
    # World [10, 0] projects to Image [u, v]: [3239.38, 360.]
    # World [5, 0]  projects to Image [u, v]: [2254.76, 360.]
    # This means World X -> Image u (horizontal), and World Y -> Image v (vertical)
    # Wait, the X axis is forward? If X=10 maps to u=3239, X=5 maps to u=2254
    # The test image is 1280x720. u=3239 is way off the screen.
    # The default CARLA view is Z forward, X right, Y down.
    # Our H matrix uses X and Y as the ground plane.
    # Let's map World (X right, Y forward). Wait, if X=10 is u=3239, then X is mapping to the right (u).
    # And Y maps to v (downwards).
    
    # The H matrix maps (X_world, Y_world, 1) -> (u, v, scale)
    # Based on the debug prints:
    # World [X=0, Y=5] -> Image [u=1270, v=1344]
    # World [X=0, Y=10] -> Image [u=1270, v=2329]
    # So Y in World maps to v (which goes down). This means as Y increases, v increases.
    # But wait, image height is 720. So v=1344 or 2329 is completely off the bottom of the screen!
    # Ah! The original calc script assumes Z is ground plane, but what are the camera extrinsics?
    # pitch = 0 (looking straight ahead). height = 3.25m.
    # The default coordinate system in CARLA for camera:
    # +X is Forward
    # +Y is Right
    # +Z is Up
    
    # Wait, euler_to_rot_matrix in calc_img_to_world uses yaw(Z), pitch(Y), roll(X).
    # If pitch = 0, R is identity.
    # Camera X is looking into the screen, Y is down, Z is right? Or X is right, Y is down, Z forward?
    # Typically in computer vision: X right, Y down, Z forward.
    # If R is identity, World = Camera. So World X=right, Y=down, Z=forward.
    # But ground plane is Z=0 in the calculate script.
    # "Assuming ground is Z=0 in world coordinates:"
    # This means the Ground is the X-Y plane.
    # So X is right, Y is down. But ground is horizontal!
    # If Y is down, and Z is forward... Wait, if camera looks forward (+Z), then ground should be Y = height.
    # But calc_img_to_world.py assumes Ground is Z=0!
    # Let's see: t = [x_offset, 0, height].
    # The calc script says: X_cam = R * [X, Y, 0] + t
    # which means World X -> Cam X (right), World Y -> Cam Y (down).
    # This means the "ground" points we are mapping are on a vertical wall in front of the camera (Z_cam = height).
    # This is WRONG! A ground plane should have Y_cam = height, or Z_world = 0 mapped properly.
    
    # Let's fix this in the IPM script directly, we can compute the correct Homography here.
    # Camera intrinsic K (scaled to 1280x720)
    fx = 960.0 * scale_x
    fy = 960.0 * scale_y
    cx = 960.0 * scale_x
    cy = 540.0 * scale_y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # We want ground to be World Z=0.
    # Camera is at (x=x_offset, y=0, z=height) in World.
    # Let World coordinate system be: X right, Y forward, Z up.
    # Camera coordinate system: X right, Y down, Z forward.
    
    # Let's calculate the correct M_world_to_cam mapping a point P_world to P_cam.
    # P_cam = R_ext * (P_world - T_ext)
    # T_ext is camera position in world: [0, 0, 3.25] (assuming x_offset=0 for simplicity, or [-3.2, 0, 3.25])
    # Camera looks along World +Y. So Cam Z = World Y.
    # Camera Right is Cam X = World X.
    # Camera Down is Cam Y = World -Z.
    
    height = 3.25
    x_offset = 3.2 # Camera is offset forward in vehicle geometry. Let's just say Vehicle X is forward.
    # In CARLA, Vehicle X is forward, Y is right, Z is up.
    # So Camera position: X = 3.2, Y = 0, Z = 3.25
    # Camera orientation: looks forward (+X_veh), right is +Y_veh, down is -Z_veh.
    # In CV Camera coords: Z_cam is forward (+X_veh). X_cam is right (+Y_veh). Y_cam is down (-Z_veh).
    
    # So given a point P_world = [X_veh, Y_veh, Z_veh]^T
    # P_cam = [ Y_veh, -(Z_veh - 3.25), X_veh - 3.2 ]^T
    # Wait, Ground is Z_veh = 0.
    # So for a ground point [X_veh, Y_veh, 0]^T
    # P_cam = [ Y_veh, 3.25, X_veh - 3.2 ]^T
    
    # We want a Homography H that maps [X_veh, Y_veh, 1]^T -> [u, v, scale]^T
    # P_cam = [0, 1, 0] * X_veh + [1, 0, 0] * Y_veh + [0, 3.25, -3.2] * 1
    #       = [Y_veh, 3.25, X_veh - 3.2]^T
    # Wait: Cam X = Y_veh, Cam Y = 3.25, Cam Z = X_veh - 3.2
    
    # H = K * [ [0, 1, 0], [0, 0, 3.25], [1, 0, -3.2] ]
    H_world_to_img = K @ np.array([
        [0, 1, 0],
        [0, 0, 3.25],
        [1, 0, -3.2]
    ])
    
    # Now we have correct H mapping Vehicle Ground (X_veh forward, Y_veh right) to Image!
    
    # Top-down image (u,v):
    resolution = 20 # px/m
    W = int(40 * resolution) # Y_veh from -20 to 20
    H_img = int(50 * resolution) # X_veh from 0 to 50
    
    # top-down (u,v) to World (X_veh, Y_veh)
    # u = 0..W correspond to Y_veh from -20 to 20 -> Y_veh = (u - W/2) / resolution
    # v = H..0 correspond to X_veh from 0 to 50 -> X_veh = (H_img - v) / resolution 
    
    M_topdown_to_world = np.array([
        [0, -1.0/resolution, H_img/resolution],   # X_veh
        [1.0/resolution, 0, -W/(2.0*resolution)], # Y_veh
        [0, 0, 1]
    ])
    
    M_topdown_to_img = H_world_to_img @ M_topdown_to_world
    
    topdown = cv2.warpPerspective(img, M_topdown_to_img, (W, H_img), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    
    cv2.imwrite(args.output, topdown)
    print(f"Topdown image saved to {args.output}")

if __name__ == '__main__':
    main()
