from typing import TYPE_CHECKING
import random
from dataclasses import dataclass

import numpy as np
from numba import njit

from .utils import normalize
from .constants import NUMBA

if TYPE_CHECKING:
    from .surfaces import Material


@dataclass
class Hit:
    t: float
    normal: np.ndarray
    material: "Material"


class Ray:
    def __init__(self, origin: np.ndarray, dir: np.ndarray, pixel_coords: tuple):
        self.origin = origin
        self.dir = normalize(dir)
        self.pixel_coords = pixel_coords
        self.color = np.array([1.0, 1, 1])
        self.luminance = 0.0
        self.hits = 0

    def hit(self, hit: Hit):
        self.hits += 1
        self.luminance += hit.material.luminance

        if NUMBA:
            self.origin, self.dir, self.color = hit_optimized(
                self.origin,
                self.dir,
                self.color,
                hit.material.color,
                hit.t,
                hit.normal,
                hit.material.reflectivity,
            )
        else:
            self.color = self.color * hit.material.color
            self.origin = self.origin + hit.t * self.dir
            if hit.material.reflectivity > random.random():
                self.dir = self.dir - 2 * self.dir.dot(hit.normal) * hit.normal
            else:
                self.color = self.color * hit.material.color
                random_dir = hit.normal + np.random.normal(0, 1, (3))
                if random_dir.dot(hit.normal) < 0:
                    random_dir = -random_dir
                self.dir = random_dir
            self.dir = normalize(self.dir)


@njit
def hit_optimized(origin, dir, color, hit_color, t, normal, reflectivity):
    origin = origin + t * dir
    if reflectivity > random.random():
        dir = dir - 2 * np.dot(dir, normal) * normal
    else:
        color = color * hit_color
        random_dir = normal + np.random.normal(0, 1, 3)
        if np.dot(random_dir, normal) < 0:
            random_dir = -random_dir
        dir = random_dir
    dir = dir / (np.linalg.norm(dir) + 1e-6)
    return origin, dir, color
