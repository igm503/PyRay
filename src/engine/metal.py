from typing import TYPE_CHECKING

import metalcompute as mc
import numpy as np

from .metal_code import metal_code
from ..types import MetalTypes

if TYPE_CHECKING:
    from view import View
    from .surfaces import Sphere, Triangle

class MetalTracer:
    def __init__(self):
        self.device = mc.Device()
        self.metal_library = self.device.kernel(metal_code)

    def render(
        self,
        view: "View",
        spheres: list["Sphere"],
        triangles: list["Triangle"],
        num_rays: int,
        max_bounces: int,
        exposure: float,
    ):
        metal_spheres, metal_triangles = self.surfaces_to_numpy(spheres, triangles)
        img = self.trace_rays(
            view,
            metal_spheres,
            metal_triangles,
            len(spheres),
            len(triangles),
            max_bounces,
            num_rays,
        )
        final_img = self.tone_map(img, exposure, (view.width, view.height))
        return final_img.reshape(view.height, view.width, 3).astype(np.uint8)

    def surfaces_to_numpy(self, spheres: list["Sphere"], triangles: list["Triangle"]):
        metal_spheres = np.zeros(max(len(spheres), 1), dtype=MetalTypes.sphere_dtype)
        for i, sphere in enumerate(spheres):
            metal_spheres[i] = sphere.to_numpy()
        numpy_triangles = np.zeros(max(len(triangles), 1), dtype=MetalTypes.triangle_dtype)
        for i, triangle in enumerate(triangles):
            numpy_triangles[i] = triangle.to_numpy()
        return metal_spheres, numpy_triangles

    def trace_rays(
        self,
        view,
        spheres,
        triangles,
        num_spheres,
        num_triangles,
        num_bounces,
        num_rays,
    ):
        random_seed = np.random.randint(0, 2**16 - 1, dtype=np.int32)
        buffers = [
            self.device.buffer(data)
            for data in [
                view.to_numpy(),
                spheres,
                triangles,
                np.array([num_spheres], dtype=np.int32),
                np.array([num_triangles], dtype=np.int32),
                np.array([num_bounces], dtype=np.int32),
                np.array([num_rays], dtype=np.int32),
                random_seed,
            ]
        ]
        image_buffer = self.device.buffer(
            view.width * view.height * 3 * np.dtype(np.float32).itemsize
        )
        buffers.append(image_buffer)
        num_threads = view.width * view.height
        self.run_kernel("trace_rays", num_threads, buffers)
        return np.frombuffer(image_buffer, dtype=np.float32)

    def tone_map(self, hdr_image, exposure, resolution):
        buffers = [
            self.device.buffer(data)
            for data in [
                hdr_image,
                np.array([exposure], dtype=np.float32),
            ]
        ]
        width, height = resolution
        image_buffer = self.device.buffer(width * height * 3 * np.dtype(np.float32).itemsize)
        buffers.append(image_buffer)
        self.run_kernel("tone_map", width * height, buffers)
        return np.frombuffer(image_buffer, dtype=np.float32)

    def run_kernel(self, name, call_count, buffers):
        kernel_fn = self.metal_library.function(name)
        kernel_fn(call_count, *buffers)
