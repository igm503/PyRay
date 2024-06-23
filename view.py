import math

import numpy as np


class View:
    def __init__(
        self, origin: np.ndarray, dir: np.ndarray, width: int, height: int, fov: float
    ):
        self.origin = origin
        self.dir = dir / np.linalg.norm(dir)
        self.width = width
        self.height = height
        self.fov = math.radians(fov)
        left_dir = -np.cross(self.dir, np.array([0, 0, 1]))
        self.left_dir = left_dir / np.linalg.norm(left_dir)

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
        x_y_magnitude = np.sqrt(self.dir[0] ** 2 + self.dir[1] ** 2)
        theta = np.arctan2(self.dir[1], self.dir[0])
        new_x_y = np.array([np.cos(theta + d_theta), np.sin(theta + d_theta), 0])
        new_x_y = (new_x_y / np.linalg.norm(new_x_y)) * x_y_magnitude
        self.dir[0:2] = new_x_y[0:2]
        left_dir = -np.cross(self.dir, np.array([0, 0, 1]))
        self.left_dir = left_dir / np.linalg.norm(left_dir)

    def look_up(self):
        self.change_dir_vertical(0.1)

    def look_down(self):
        self.change_dir_vertical(-0.1)

    def change_dir_vertical(self, d_theta):
        horizontal_vec = np.array([self.dir[0], self.dir[1], 0]) 
        horizontal_comp = np.sqrt(self.dir[0] ** 2 + self.dir[1] ** 2)
        horizontal_normal = horizontal_vec / horizontal_comp
        theta = np.arctan2(self.dir[2], horizontal_comp)
        new_theta = theta + d_theta
        new_theta = np.clip(new_theta, -np.pi / 2 + 0.01, np.pi / 2)
        new_z_dir = np.sin(new_theta)
        x_y_scale = np.cos(new_theta)
        self.dir = np.array([horizontal_normal[0] * x_y_scale, horizontal_normal[1] * x_y_scale, new_z_dir])
        self.dir = self.dir / np.linalg.norm(self.dir)
