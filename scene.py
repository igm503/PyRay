import time
import math

import numpy as np
import cv2
from tqdm import tqdm

from ray import Ray
from surface import Surface, Hit, Material
from triangle import Triangle
from sphere import Sphere
from view import View
from utils import normalize
from metal import MetalTracer

sun_hit = Hit(
    t=1,
    normal=np.array([0, 0.0, 0]),
    material=Material(color=np.array([1, 1, 1]), reflectivity=0.0, luminance=1.0),
)


def get_sky_hit(ray: Ray):
    ground_plane_component = ray.dir[0] ** 2 + ray.dir[1] ** 2
    color = np.array([1, 0.9, 0.5]) + np.array([0, 0.1, 0.5]) * ground_plane_component
    return Hit(
        t=1,
        normal=np.array([0, 0.0, 0]),
        material=Material(color=color, reflectivity=0.0, luminance=0.5),
    )


class Scene:
    def __init__(self, surfaces: list[Surface]):
        self.triangles = [surface for surface in surfaces if isinstance(surface, Triangle)]
        self.spheres = [surface for surface in surfaces if isinstance(surface, Sphere)]
        self.surfaces = surfaces
        self.tracer = MetalTracer()

    def get_rays(self, view: View, num_rays: int):
        pixel_unit = (2 * math.tan(view.fov / 2)) / view.width
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

    def static_render(
        self,
        view: View,
        num_rays: int,
        max_bounces: int,
        save_dir: str | None = None,
    ):
        current_img = None
        img = np.zeros((view.height, view.width, 3))
        for i in tqdm(range(num_rays)):
            for ray in self.get_rays(view, 1):
                self.trace_ray(ray, max_bounces)
                if ray.hits > 0:
                    img[ray.pixel_coords[::-1]] += ray.color * min(ray.luminance, 1.0) * 255
            current_img = img / (i + 1)
            current_img = np.clip(current_img, 0, 255).astype(np.uint8)
            print(f"Rendering... {i+1}/{num_rays}")
            if save_dir is not None:
                cv2.imwrite(f"{save_dir}/output_{i}.png", current_img)
            cv2.imshow("image", current_img)
            cv2.waitKey(1)
        return current_img

    def surfaces_to_numpy(self):
        if not self.spheres:
            self.numpy_spheres = np.zeros(1, dtype=MetalTracer.sphere_dtype)
        else:
            self.numpy_spheres = np.zeros(len(self.spheres), dtype=MetalTracer.sphere_dtype)
        for i, sphere in enumerate(self.spheres):
            self.numpy_spheres[i] = sphere.to_numpy()
        if not self.triangles:
            self.numpy_triangles = np.zeros(1, dtype=MetalTracer.triangle_dtype)
        else:
            self.numpy_triangles = np.zeros(len(self.triangles), dtype=MetalTracer.triangle_dtype)
        for i, triangle in enumerate(self.triangles):
            self.numpy_triangles[i] = triangle.to_numpy()

    def metal_render(self, view: View, num_rays: int, max_bounces: int):
        self.surfaces_to_numpy()
        img = self.tracer.trace_rays(
            view,
            self.numpy_spheres,
            self.numpy_triangles,
            len(self.spheres),
            len(self.triangles),
            max_bounces,
            num_rays,
        )
        final_img = self.tracer.tone_map(img, 5.5, (view.width, view.height))
        return final_img.reshape(view.height, view.width, 3).astype(np.uint8)

    def cpu_render(self, view: View, num_rays: int, max_bounces: int):
        img = np.zeros((view.height, view.width, 3))
        for ray in self.get_rays(view, num_rays):
            self.trace_ray(ray, max_bounces)
            if ray.hits > 0:
                img[ray.pixel_coords[::-1]] += ray.color * min(ray.luminance, 1.0) * 255
        img /= num_rays
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def trace_ray(self, ray: Ray, max_bounces: int):
        closest_hit = None
        for _ in range(max_bounces):
            for surface in self.surfaces:
                hit = surface.check_hit(ray)
                if hit is not None and (closest_hit is None or hit.t < closest_hit.t):
                    closest_hit = hit
            if closest_hit is None:
                if ray.dir.dot(np.array([0, 0, 1])) > 0.98:
                    ray.hit(sun_hit)
                else:
                    ray.hit(get_sky_hit(ray))
                break
            else:
                ray.hit(closest_hit)
            closest_hit = None
