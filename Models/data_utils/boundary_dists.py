import numpy as np

class BoundaryDists:
    def __init__(self, image, i):
        self.image = image
        self.num_points = i
        self.height, self.width = image.shape[:2]
        self.drivable_label = 2

    def get_boundary_dists(self):
        cols = np.linspace(0, self.width - 1, self.num_points, dtype=int)
        normalized_distances = []
        for col in cols:
            boundary_found = False
            for row in range(self.height - 1, -1, -1):
                if self.image[row, col] != self.drivable_label:
                    normalized_distance = (self.height - 1 - row) / self.height
                    normalized_distances.append(normalized_distance)
                    boundary_found = True
                    break

            if not boundary_found:
                normalized_distances.append(1.0)

        return normalized_distances