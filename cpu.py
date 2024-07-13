import numpy as np

from view import View
from ray import Ray
from sphere import Sphere
from triangle import Triangle
from surface import Surface, Hit, Material

from utils import normalize

SUN = np.array((1.0, 0.68, 0.26))
WHITE = np.array((1.0, 1.0, 1.0))
SKY = np.array((0.53, 0.81, 0.92))


class CPUTracer:
    def render(
        self,
        view: View,
        spheres: list[Sphere],
        triangles: list[Triangle],
        num_rays: int,
        max_bounces: int,
        exposure: float
    ):
        img = np.zeros((view.height, view.width, 3))
        for ray in self.get_rays(view, num_rays):
            self.trace_ray(ray, spheres + triangles, max_bounces)
            if ray.hits > 0:
                img[ray.pixel_coords[::-1]] += ray.color * ray.luminance
        img /= num_rays
        img = self.tone_map(img, exposure)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def get_rays(self, view: View, num_rays: int):
        pixel_unit = (2 * np.tan(view.fov / 2)) / view.width
        left_dir = normalize(np.cross(np.array([0, 0, 1]), view.dir)) * pixel_unit
        up_dir = normalize(np.cross(view.dir, left_dir)) * pixel_unit
        top_left = view.dir + view.width / 2 * left_dir + view.height / 2 * up_dir

        rays = []
        for x in range(view.width):
            for y in range(view.height):
                dir = normalize(top_left - x * left_dir - y * up_dir)
                for _ in range(num_rays):
                    rays.append(Ray(view.origin, dir, (x, y)))

        return rays

    def trace_ray(self, ray: Ray, surfaces: list[Surface], max_bounces: int):
        closest_hit = None
        for _ in range(max_bounces):
            for surface in surfaces:
                hit = surface.check_hit(ray)
                if hit is not None and (closest_hit is None or hit.t < closest_hit.t):
                    closest_hit = hit
            if closest_hit is None:
                hit = self.get_environment_hit(ray)
                ray.hit(hit)
                break
            else:
                ray.hit(closest_hit)
            closest_hit = None

    def get_environment_hit(self, ray: Ray):
        if ray.dir[-1] > 0.99:
            scale = (ray.dir[-1] - 0.98) / 0.02
            return Hit(
                t=1,
                normal=np.array([0, 0.0, 0]),
                material=Material(
                    color=scale * WHITE + (1 - scale) * SUN,
                    reflectivity=0.0,
                    luminance=1.0,
                ),
            )
        else:
            return Hit(
                t=1,
                normal=np.array([0, 0.0, 0]),
                material=Material(color=SKY, reflectivity=0.0, luminance=0.5),
            )

    def tone_map(self, img: np.ndarray, exposure: float):
        img = 1 - np.exp(-img * exposure)
        img = (img * 255).astype(np.uint8)
        return img
