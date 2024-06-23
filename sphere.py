import numpy as np
from numba import njit

from ray import Ray
from surface import Surface, Hit
from constants import NUMBA


class Sphere(Surface):
    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        color: np.ndarray,
        material: str,
        luminance: float = 0,
    ):
        self.center = center
        self.radius = radius
        self.color = color
        self.material = material
        self.luminance = luminance

    def check_hit(self, ray: Ray):
        if NUMBA:
            did_hit, normal, t = check_hit_jit(
                self.center, self.radius, ray.origin, ray.dir
            )
        else:
            did_hit = False
            ray_offset_origin = ray.origin - self.center

            # b = 2 * ray.dir.dot(ray_offset_origin)
            b = 2 * (
                ray.dir[0] * ray_offset_origin[0]
                + ray.dir[1] * ray_offset_origin[1]
                + ray.dir[2] * ray_offset_origin[2]
            )
            # c = ray_offset_origin.dot(ray_offset_origin) - self.radius ** 2
            c = (
                ray_offset_origin[0] ** 2
                + ray_offset_origin[1] ** 2
                + ray_offset_origin[2] ** 2
                - self.radius**2
            )
            discriminant = b**2 - 4 * c

            if discriminant > 0:
                t = (-b - np.sqrt(discriminant)) / 2
                if t > 0:
                    did_hit = True
                    hit_point = ray.origin + t * ray.dir
                    normal = (hit_point - self.center) / self.radius
        if did_hit:
            return Hit(
                t=t,
                normal=normal,
                color=self.color,
                material=self.material,
                luminance=self.luminance,
            )

        return None


@njit
def check_hit_jit(center, radius, ray_origin, ray_dir):
    ray_offset_origin = ray_origin - center

    # b = 2 * ray.dir.dot(ray_offset_origin)
    b = 2 * (
        ray_dir[0] * ray_offset_origin[0]
        + ray_dir[1] * ray_offset_origin[1]
        + ray_dir[2] * ray_offset_origin[2]
    )
    # c = ray_offset_origin.dot(ray_offset_origin) - self.radius ** 2
    c = (
        ray_offset_origin[0] ** 2
        + ray_offset_origin[1] ** 2
        + ray_offset_origin[2] ** 2
        - radius**2
    )
    discriminant = b**2 - 4 * c

    if discriminant > 0:
        t = (-b - np.sqrt(discriminant)) / 2
        if t > 0:
            hit_point = ray_origin + t * ray_dir
            normal = (hit_point - center) / radius
            return True, normal, t
    return None, None, None
