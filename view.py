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
        self.fov = math.radians(fov)
        self.reset_rays()


    def reset_rays(self):
        self.rays = []
        projection_center = self.origin + self.dir
        left_dir = -np.cross(self.dir, np.array([0, 0, 1]))
        self.left_dir = left_dir / np.linalg.norm(left_dir)
        up_dir = np.cross(self.dir, left_dir)
        up_dir = up_dir / np.linalg.norm(up_dir)
        pixel_unit = (2 * math.tan(self.fov / 2)) / self.width
        top_left = projection_center + math.tan(self.fov / 2) * left_dir + (pixel_unit * self.height / 2) * up_dir

        for x in range(self.width):
            for y in range(self.height):
                point = top_left - x * pixel_unit * left_dir - y * pixel_unit * up_dir
                dir = point - self.origin
                self.rays.append(Ray(self.origin, dir, (x, y)))

    def forward(self):
        self.origin = self.origin + self.dir * 0.1

    def backward(self):
        self.origin = self.origin - self.dir * 0.1

    def left(self):
        self.origin = self.origin + self.left_dir * 0.1

    def right(self):
        self.origin = self.origin - self.left_dir * 0.1

    def look_left(self):
        self.change_dir_horizontal(0.1)

    def look_right(self):
        self.change_dir_horizontal(-0.1)

    def change_dir_horizontal(self, d_theta):
        print(self.dir)
        x_y_magnitude = np.sqrt(self.dir[0] ** 2 + self.dir[1] ** 2)
        theta = np.arctan2(self.dir[1], self.dir[0])
        new_ratio = np.tan(theta + d_theta)
        new_x_y = np.array([1, new_ratio, 0])
        new_x_y = (new_x_y / np.linalg.norm(new_x_y)) * x_y_magnitude
        self.dir[0:2] = new_x_y[0:2] 
        print(self.dir)

    def look_up(self):
        self.change_dir_vertical(0.1)

    def look_down(self):
        self.change_dir_vertical(-0.1)

    def change_dir_vertical(self, d_theta):
        horizontal_comp = np.sqrt(self.dir[0] ** 2 + self.dir[1] ** 2)
        theta = np.arctan2(self.dir[2], horizontal_comp)
        new_ratio = np.tan(theta + d_theta)
        new_z_dir = horizontal_comp * new_ratio
        self.dir = np.array([self.dir[0], self.dir[1], new_z_dir])
        self.dir = self.dir / np.linalg.norm(self.dir)







