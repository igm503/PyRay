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
        closest_hit = None
        for _ in range(max_bounces):
            for surface in self.surfaces:
                # check_start = time.time()
                hit = surface.check_hit(ray)
                # check_time += time.time() - check_start
                # print(hit)
                # print(closest_hit)
                if hit is not None and (closest_hit is None or hit.t < closest_hit.t):
                    closest_hit = hit
                    # print("new closest hit")
                    # print(closest_hit)
            if closest_hit is None:
                if ray.dir.dot(np.array([0, 0, 1])) > 0:
                    if ray.dir.dot(np.array([0, 0, 1])) > .95:
                        ray.hit(sun_hit)
                    ray.hit(sky_hit)
                else:
                    ray.hit(grass_hit)
                break
            else:
                # hit_start = time.time()
                ray.hit(closest_hit)
                # hit_time += time.time() - hit_start
            closest_hit = None
        # print("hit:", hit_time, "check:", check_time, "total:", time.time() - start)
