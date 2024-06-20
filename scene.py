import time

import numpy as np

from ray import Ray
from surface import Surface, Hit
from view import View

sky_hit = Hit(color=np.array([100, 100, 0.0]), normal=np.array([0, 0.0, 0]), t=1, material="diffuse")
grass_hit = Hit(color=np.array([0.0, 100, 0.0]), normal=np.array([0, 0.0, 0]), t=1, material="diffuse")
sun_hit = Hit(color=np.array([255, 255, 255]), normal=np.array([0, 0.0, 0]), t=1, material="diffuse")

class Scene:
    def __init__(self, surfaces: list[Surface]):
        self.surfaces = surfaces

    def render(self, view: View, max_bounces: int):
        start = time.time()
        print(len(view.rays))
        for ray in view.rays:
            self.trace_ray(ray, max_bounces)
        trace = time.time()
        img = np.zeros((view.height, view.width, 3))
        for ray in view.rays:
            img[ray.pixel_coords[::-1]] = ray.color
        draw = time.time()
        print("trace:", trace - start, "draw:", draw - trace)
        return img

    def trace_ray(self, ray: Ray, max_bounces: int):
        # check_time = 0
        # hit_time = 0
        # start = time.time()
        closest_hit = Hit()
        for _ in range(max_bounces):
            for surface in self.surfaces:
                # check_start = time.time()
                hit, color, normal, t = surface.check_hit(ray)
                # check_time += time.time() - check_start
                if hit and (closest_hit.t is None or t < closest_hit.t):
                    closest_hit.color = color
                    closest_hit.normal = normal
                    closest_hit.material = surface.material
                    closest_hit.t = t
            if closest_hit.t is None:
                if ray.dir.dot(np.array([0, 0, 1])) > 0:
                    if ray.dir.dot(np.array([0, 0, 1])) > .95:
                        ray.hit(sun_hit)
                    ray.hit(sky_hit)
                else:
                    ray.hit(grass_hit)
                break
            # hit_start = time.time()
            ray.hit(closest_hit)
            # hit_time += time.time() - hit_start
            closest_hit.t = None
        # print("hit:", hit_time, "check:", check_time, "total:", time.time() - start)
