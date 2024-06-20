import math

import numpy as np

from ray import Ray


class View:
    def __init__(
        self, origin: np.ndarray, dir: np.ndarray, width: int, height: int, fov: float
    ):
        self.origin = origin
        self.dir = dir / np.linalg.norm(dir)
        self.width = width
        self.height = height
        fov = math.radians(fov)

        self.rays = []
        projection_center = self.origin + self.dir
        print(-np.cross(self.dir, np.array([0, 0, 1])))
        left_dir = -np.cross(self.dir, np.array([0, 0, 1]))
        print(np.linalg.norm(left_dir))
        left_dir = left_dir / np.linalg.norm(left_dir)
        print(left_dir)
        up_dir = np.cross(self.dir, left_dir)
        up_dir = up_dir / np.linalg.norm(up_dir)
        print(up_dir)
        pixel_unit = (2 * math.tan(fov / 2)) / self.width
        top_left = projection_center + math.tan(fov / 2) * left_dir + (pixel_unit * self.height / 2) * up_dir

        for x in range(self.width):
            for y in range(self.height):
                point = top_left - x * pixel_unit * left_dir - y * pixel_unit * up_dir
                dir = point - self.origin
                self.rays.append(Ray(self.origin, dir, (x, y)))

    def reset_rays(self):
        for ray in self.rays:
            ray.reset(self.origin)
        print(self.rays[0].origin, self.rays[0].dir)
        print(self.rays[-1].origin, self.rays[-1].dir)
