import numpy as np
from numba import njit

from surface import Hit, Material
from utils import normalize
from constants import NUMBA


class Ray:
    def __init__(self, origin: np.ndarray, dir: np.ndarray, pixel_coords: tuple):
        self.origin = origin
        self.dir = normalize(dir)
        self.pixel_coords = pixel_coords
        self.color = np.array([1.0, 1, 1])
        self.luminance = 0
        self.hits = 0

    def hit(self, hit: Hit):
        self.hits += 1
        self.luminance += hit.luminance

        if NUMBA:
            self.origin, self.dir, self.color = hit_optimized(
                self.origin,
                self.dir,
                self.color,
                hit.color,
                hit.t,
                hit.normal,
                hit.material,
            )
        else:
            if self.color is None:
                self.color = hit.color
            else:
                self.color = np.minimum(self.color, hit.color)

            self.origin = self.origin + hit.t * self.dir

            if hit.material == Material.DIFFUSE:
                random_dir = np.random.normal(0, 1, (3))
                if random_dir.dot(hit.normal) < 0:
                    random_dir = -random_dir
                self.dir = random_dir
            elif hit.material == Material.SPECULAR:
                self.dir = self.dir - 2 * self.dir.dot(hit.normal) * hit.normal
            else:
                raise ValueError("Invalid material:", hit.material)

            self.dir = self.dir / np.linalg.norm(self.dir)


@njit
def hit_optimized(origin, dir, color, hit_color, hit_t, hit_normal, hit_material):
    origin = origin + hit_t * dir
    if hit_material == "diffuse":
        color = color * hit_color
        random_dir = np.random.normal(0, 1, 3)
        if np.dot(random_dir, hit_normal) < 0:
            random_dir = -random_dir
        dir = random_dir
    elif hit_material == "specular":
        dir = dir - 2 * np.dot(dir, hit_normal) * hit_normal
    else:
        raise ValueError("Invalid material:", hit_material)
    dir = dir / (np.linalg.norm(dir) + 1e-6)
    return origin, dir, color
