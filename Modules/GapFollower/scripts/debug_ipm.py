import cv2
import yaml
import numpy as np

def main():
    image_path = '/home/autoware/autoware.off-road/Modules/GapFollower/test/test.png'
    yaml_path = '/home/autoware/autoware.off-road/Modules/GapFollower/test/image_to_world_transform.yaml'
    
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load")
        return
    h_img, w_img = img.shape[:2]
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    H_world_to_img_1920 = np.array(data['homography_world_to_img'])
    
    scale_x = w_img / 1920.0
    scale_y = h_img / 1080.0
    M_scale = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
    
    H_world_to_img = M_scale @ H_world_to_img_1920
    
    # We want to map points in front of the vehicle (e.g. 5m to 20m)
    # The default coordinate system is typically:
    # Camera frame: Z forward, X right, Y down
    # But how was R calculated? euler_to_rot_matrix(pitch, yaw, roll)
    # The calc_img_to_world script says:
    # Assumes rotation order: yaw (Z), pitch (Y), roll (X)
    # Rz.dot(Ry).dot(Rx)
    # Extrinsic setup:
    # t = [x_offset, 0, height] => this means the camera position in world is [x_offset, 0, height]
    # In world frame, if Z is up, then Y is forward, X is right?
    # Let's map world X = 0 (center), Y = 10 (forward), Z = 0 (ground)
    
    print("Testing world points (X, Y) -> image (u, v):")
    for y in [5, 10, 20]:
        for x in [-5, 0, 5]:
            pt_world = np.array([x, y, 1.0])
            pt_img = H_world_to_img @ pt_world
            pt_img = pt_img / pt_img[2]
            print(f"World [X={x}, Y={y}] projects to Image [u, v]: {pt_img[:2]}")

if __name__ == '__main__':
    main()
