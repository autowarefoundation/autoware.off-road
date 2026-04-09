import numpy as np

class RadialContourDists:
    def __init__(self, image, num_rays=37, ray_slice_dist=10, max_search_dist=460):
        self.image = image
        self.num_rays = num_rays
        self.ray_slice_dist = ray_slice_dist
        self.max_search_dist = max_search_dist
        self.height, self.width = image.shape[:2]
        self.drivable_label = 1

    def get_contour_dists(self):
        # Scan from Left (Pi) to Right (0) to match original Left-Right scan order
        angles = np.linspace(np.pi, 0, self.num_rays)
        
        boundary_indices = []
        
        # Bottom center of the image
        start_r = self.height - 1
        start_c = self.width // 2
        
        for angle in angles:
            found = False
            # Calculate number of steps
            num_steps = int(self.max_search_dist / self.ray_slice_dist)
            
            for i in range(num_steps):
                dist = i * self.ray_slice_dist
                # Calculate coordinates
                # angle 0 (right): r same, c increases
                # angle pi/2 (up): r decreases, c same
                current_r = start_r - dist * np.sin(angle)
                current_c = start_c + dist * np.cos(angle)
                
                # Check bounds
                if not (0 <= current_r < self.height and 0 <= current_c < self.width):
                    # Out of bounds - treat as boundary
                    boundary_indices.append(i)
                    found = True
                    break
                
                # Check drivable label
                # Nearest neighbor interpolation for pixel access
                r_idx = int(round(current_r))
                c_idx = int(round(current_c))
                
                if self.image[r_idx, c_idx] != self.drivable_label:
                    boundary_indices.append(i)
                    found = True
                    break
            
            if not found:
                boundary_indices.append(num_steps - 1)
                
        return boundary_indices