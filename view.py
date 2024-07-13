import math

import numpy as np

from utils import normalize
from metal import MetalTypes

class View:
    def __init__(
        self, origin: np.ndarray, dir: np.ndarray, width: int, height: int, fov: float
    ):
        self.origin = origin
        self.dir = normalize(dir)
        self.fov = math.radians(fov)
        self.width = width
        self.height = height
        self.left_dir = normalize(np.cross(np.array([0, 0, 1]), self.dir))

        self.move_speed = 3
        self.cam_speed = .1

    def to_numpy(self):
        pixel_unit = (2.0 * math.tan(self.fov / 2.0)) / self.width
        right_dir = -self.left_dir * pixel_unit
        down_dir = -normalize(np.cross(self.dir, self.left_dir)) * pixel_unit
        top_left_dir = self.dir - (self.width / 2) * right_dir - (self.height / 2) * down_dir
        return np.array(
            (self.origin, top_left_dir, right_dir, down_dir, self.width, self.height),
            dtype=MetalTypes.view_dtype,
        )

    def forward(self):
        self.origin = self.origin + self.dir * self.move_speed

    def back(self):
        self.origin = self.origin - self.dir * self.move_speed

    def left(self):
        self.origin = self.origin + self.left_dir * self.move_speed

    def right(self):
        self.origin = self.origin - self.left_dir * self.move_speed

    def look_left(self):
        self.change_dir_horizontal(self.cam_speed)

    def look_right(self):
        self.change_dir_horizontal(-self.cam_speed)

    def change_dir_horizontal(self, d_theta):
        x_y_magnitude = np.sqrt(self.dir[0] ** 2 + self.dir[1] ** 2)
        theta = np.arctan2(self.dir[1], self.dir[0])
        new_x_y = np.array([np.cos(theta + d_theta), np.sin(theta + d_theta), 0])
        new_x_y = normalize(new_x_y) * x_y_magnitude
        self.dir[0:2] = new_x_y[0:2]
        self.left_dir = normalize(np.cross(np.array([0, 0, 1]), self.dir))

    def look_up(self):
        self.change_dir_vertical(self.cam_speed)

    def look_down(self):
        self.change_dir_vertical(-self.cam_speed)

    def change_dir_vertical(self, d_theta):
        horizontal_vec = np.array([self.dir[0], self.dir[1], 0])
        horizontal_norm = np.linalg.norm(horizontal_vec)
        horizontal_normal = normalize(horizontal_vec)
        theta = np.arctan2(self.dir[2], horizontal_norm)
        new_theta = np.clip(theta + d_theta, -np.pi / 2 + 0.01, np.pi / 2)
        new_z_dir = np.sin(new_theta)
        x_y_scale = np.cos(new_theta)
        new_dir = np.array(
            [
                horizontal_normal[0] * x_y_scale,
                horizontal_normal[1] * x_y_scale,
                new_z_dir,
            ]
        )
        self.dir = normalize(new_dir)
