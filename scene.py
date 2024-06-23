import time
import math

import numpy as np
import cv2

from ray import Ray
from surface import Surface, Hit
from view import View
from utils import normalize

sun_hit = Hit(
    color=np.array([1, 1, 1]),
    normal=np.array([0, 0.0, 0]),
    t=1,
    material="diffuse",
    luminance=1.0,
)


def get_sky_hit(ray: Ray):
    ground_plane_component = ray.dir[0] ** 2 + ray.dir[1] ** 2
    color = np.array([1, 0.9, 0.5]) + np.array([0, 0.1, 0.5]) * ground_plane_component
    return Hit(
        color=color,
        normal=np.array([0, 0.0, 0]),
        t=1,
        material="diffuse",
        luminance=0.5,
    )


class Scene:
    def __init__(self, surfaces: list[Surface]):
        self.surfaces = surfaces

    def get_rays(self, view: View, num_rays: int):
        rays = []
        projection_center = view.origin + view.dir
        self.left_dir = left_dir = normalize(np.cross(np.array([0, 0, 1]), view.dir))
        up_dir = normalize(np.cross(view.dir, left_dir))
        pixel_unit = (2 * math.tan(view.fov / 2)) / view.width
        top_left = (
            projection_center
            + math.tan(view.fov / 2) * left_dir
            + (pixel_unit * view.height / 2) * up_dir
        )
        right_dir = -left_dir
        down_dir = -up_dir

        for x in range(view.width):
            for y in range(view.height):
                point = (
                    top_left + x * pixel_unit * right_dir + y * pixel_unit * down_dir
                )
                dir = normalize(point - view.origin)
                for _ in range(num_rays):
                    rays.append(Ray(view.origin, dir, (x, y)))

        return rays

    def static_render(self, view: View, num_rays: int, max_bounces: int):
        img = np.zeros((view.height, view.width, 3))
        for i in range(num_rays):
            for ray in self.get_rays(view, 1):
                self.trace_ray(ray, max_bounces)
                if ray.hits > 0:
                    img[ray.pixel_coords[::-1]] += (
                        ray.color * min(ray.luminance, 1.0) * 255
                    )
            current_img = img / (i + 1)
            current_img = np.clip(current_img, 0, 255).astype(np.uint8)
            cv2.putText(
                current_img,
                f"Rendering... {i+1}/{num_rays}",
                (5, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("image", current_img)
            cv2.waitKey(1)

    def interactive_render(self, view: View, num_rays: int, max_bounces: int):
        img = np.zeros((view.height, view.width, 3))
        for ray in self.get_rays(view, num_rays):
            self.trace_ray(ray, max_bounces)
            if ray.hits > 0:
                img[ray.pixel_coords[::-1]] += ray.color * min(ray.luminance, 1.0) * 255
        img /= num_rays
        img = np.clip(img, 0, 255).astype(np.uint8)
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
                if hit is not None and (closest_hit is None or hit.t < closest_hit.t):
                    closest_hit = hit
            if closest_hit is None:
                if ray.dir.dot(np.array([0, 0, 1.0])) > 0:
                    if ray.dir.dot(np.array([0, 0, 1])) > 0.95:
                        ray.hit(sun_hit)
                    else:
                        ray.hit(get_sky_hit(ray))
                break
            else:
                # hit_start = time.time()
                ray.hit(closest_hit)
                # hit_time += time.time() - hit_start
            closest_hit = None
        # print("hit:", hit_time, "check:", check_time, "total:", time.time() - start)
