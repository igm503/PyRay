import numpy as np
from numba import njit

from surface import Hit, Material


class Ray:
    def __init__(self, origin: np.ndarray, dir: np.ndarray, pixel_coords: tuple):
        self.origin = origin
        self.dir = dir / np.linalg.norm(dir)
        self.pixel_coords = pixel_coords
        self.color = None

        self.og_origin = origin
        self.og_dir = self.dir

    def hit(self, hit: Hit):
        if self.color is None:
            self.color = hit.color
        self.origin, self.dir, self.color = hit_optimized(self.origin, self.dir, self.color, hit.color, hit.t, hit.normal, hit.material)
        # if self.color is None:
        #     self.color = hit.color
        # else:
        #     self.color = np.minimum(self.color, hit.color)
        #
        # self.origin = self.origin + hit.t * self.dir
        #
        # if hit.material == Material.DIFFUSE:
        #     random_dir = np.random.normal(0, 1, (3))
        #     if random_dir.dot(hit.normal) < 0:
        #         random_dir = -random_dir
        #     self.dir = random_dir
        # elif hit.material == Material.SPECULAR:
        #     self.dir = self.dir - 2 * self.dir.dot(hit.normal) * hit.normal
        # else:
        #     raise ValueError("Invalid material:", hit.material)
        #
        # self.dir = self.dir / np.linalg.norm(self.dir)

    def reset(self, new_origin: np.ndarray | None = None):
        self.color = None
        self.dir = self.og_dir
        if new_origin is not None:
            self.origin = new_origin
            self.og_origin = new_origin
        else:
            self.origin = self.og_origin

@njit
def hit_optimized(origin, dir, color, hit_color, hit_t, hit_normal, hit_material):
    color = np.minimum(color, hit_color)
    
    origin = origin + hit_t * dir
    
    if hit_material == "diffuse":
        random_dir = np.random.normal(0, 1, 3)
        if np.dot(random_dir, hit_normal) < 0:
            random_dir = -random_dir
        dir = random_dir
    elif hit_material == "specular":
        dir = dir - 2 * np.dot(dir, hit_normal) * hit_normal
    else:
        raise ValueError("Invalid material:", hit_material)
    
    dir = normalize(dir)
    
    return origin, dir, color

@njit
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
