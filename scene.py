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
        material=Material(color=color, reflectivity=0.0, luminance=0.1),
    )


class Scene:
    def __init__(self, surfaces: list[Surface]):
        self.triangles = [surface for surface in surfaces if isinstance(surface, Triangle)]
        self.spheres = [surface for surface in surfaces if isinstance(surface, Sphere)]
        self.surfaces = surfaces
        self.tracer = MetalTracer()

    def get_rays_exp(self, view: View, num_rays: int):
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

        x_array = np.tile(np.arange(view.width) * pixel_unit, (view.height, 1))
        x_array = x_array.reshape(view.height, view.width, 1)
        y_array = np.repeat(np.arange(view.height) * pixel_unit, view.width)
        y_array = y_array.reshape(view.height, view.width, 1)

        ray_dirs = top_left + x_array * right_dir + y_array * down_dir - view.origin
        dir_norms = np.linalg.norm(ray_dirs, axis=2)
        ray_dirs = ray_dirs / dir_norms.reshape(view.height, view.width, 1)

        ray_origins = np.tile(view.origin, (view.height, view.width, 1))
        ray_colors = np.zeros((view.height, view.width, 3)).astype(np.float32)
        ray_luminances = np.zeros((view.height, view.width)).astype(np.float32)
        ray_hits = np.zeros((view.height, view.width)).astype(np.int32)

        return ray_origins, ray_dirs, ray_colors, ray_luminances, ray_hits

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

        x, y = np.meshgrid(np.arange(view.width), np.arange(view.height))

        ray_dirs = (
            top_left
            + x[:, :, np.newaxis] * pixel_unit * right_dir
            + y[:, :, np.newaxis] * pixel_unit * down_dir
            - view.origin
        )

        dir_norms = np.linalg.norm(ray_dirs, axis=2)
        ray_dirs /= dir_norms.reshape(view.height, view.width, 1)

        rays = np.zeros(view.width * view.height * num_rays, dtype=MetalTracer.ray_dtype)

        rays["origin"] = np.tile(view.origin, (view.width * view.height * num_rays, 1))
        rays["direction"] = np.repeat(ray_dirs, num_rays, axis=0).reshape(-1, 3)
        rays["pixel"] = np.stack((x, y), axis=2).reshape(-1, 2)
        rays["color"] = np.tile(np.array([1.0, 1, 1]), (view.width * view.height * num_rays, 1))
        rays["intensity"] = np.zeros(view.width * view.height * num_rays)

        return rays

    def static_render(
        self, view: View, num_rays: int, max_bounces: int, save_dir: str | None = None
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
            render_label = f"Rendering... {i+1}/{num_rays}"
            if save_dir is not None:
                cv2.imwrite(f"{save_dir}/output_{i}.png", current_img)
            cv2.putText(
                current_img,
                render_label,
                (5, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
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
        t1 = time.time()
        img = np.zeros((view.height, view.width, 3))
        t2 = time.time()
        print(img.size)
        print("Time to get rays:", round(t2 - t1, 3))
        img = self.tracer.trace_rays(
            view,
            self.numpy_spheres,
            self.numpy_triangles,
            len(self.spheres),
            len(self.triangles),
            max_bounces,
            num_rays,
        )
        t3 = time.time()
        print("Time to trace rays:", round(t3 - t2, 3))
        final_img = self.tracer.tone_map(img, 5.5, (view.width, view.height))
        print("Time to tone map:", round(time.time() - t3, 3))
        return final_img.reshape(view.height, view.width, 3).astype(np.uint8)

    def interactive_render(self, view: View, num_rays: int, max_bounces: int):
        img = np.zeros((view.height, view.width, 3))
        ray_origins, ray_dirs, ray_colors, ray_luminances, ray_hits = self.get_rays(view, num_rays)
        # for _ in range(max_bounces):
        ray_origins, ray_dirs, ray_colors, ray_luminances, ray_hits = self.trace_rays(
            ray_origins, ray_dirs, ray_colors, ray_luminances, ray_hits
        )
        ray_luminances = np.clip(ray_luminances, 0.9, 1)
        ray_colors = ray_colors * ray_luminances.reshape(view.height, view.width, 1) * 255
        ray_colors = np.clip(ray_colors, 0, 255).astype(np.uint8)
        return ray_colors

    def trace_rays(self, ray_origins, ray_dirs, ray_colors, ray_luminances, ray_hits):
        # check_time = 0
        # hit_time = 0
        # start = time.time()
        hit_normals = np.zeros((ray_origins.shape[0], ray_origins.shape[1], 3)).astype(np.float32)
        hit_ts = np.full((ray_origins.shape[0], ray_origins.shape[1]), np.inf).astype(np.float32)
        has_hit = np.zeros((ray_origins.shape[0], ray_origins.shape[1])).astype(np.bool)
        # new_ray_origins = np.zeros_like(ray_origins)
        # new_ray_dirs = np.zeros_like(ray_dirs)
        for surface in self.surfaces:
            did_hit, normals, t = surface.check_hit(ray_origins, ray_dirs)
            print(did_hit.sum())
            # update hit normals if did hit and t < current hit t
            hit_normals[did_hit & (t < hit_ts)] = normals[did_hit & (t < hit_ts)]
            hit_ts[did_hit & (t < hit_ts)] = t[did_hit & (t < hit_ts)]
            has_hit = has_hit | did_hit
            print(has_hit.sum())
        print()

        # new_ray_origins = ray_origins + ray_dirs * hit_ts.reshape(ray_origins.shape[0], ray_origins.shape[1], 1)
        # new_ray_dirs =

        # if closest_hit is None:
        #     if ray.dir.dot(np.array([0, 0, 1])) > 0.98:
        #         ray.hit(sun_hit)
        #     else:
        #         get_sky_hit(ray_dirs)
        #     break
        # else:
        #     ray.hit(closest_hit)
        # print(sum(has_hit))
        print(ray_colors.sum())
        ray_colors[has_hit] = np.array([1, 1, 1])
        return ray_origins, ray_dirs, ray_colors, ray_luminances, ray_hits
