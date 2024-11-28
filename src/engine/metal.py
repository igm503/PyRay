from typing import TYPE_CHECKING
from pathlib import Path

import metalcompute as mc
import numpy as np

from ..types import inputs_to_numpy

if TYPE_CHECKING:
    from view import View
    from ..surfaces import Sphere, Triangle


class MetalTracer:
    def __init__(self):
        self.device = mc.Device()

        metal_code_path = Path(__file__).parent / "metal_code.metal"
        with open(metal_code_path, "r") as f:
            metal_code = f.read()
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
        _, np_spheres, np_triangles = inputs_to_numpy(view, spheres, triangles)
        img = self.trace_rays(view, np_spheres, np_triangles, max_bounces, num_rays)
        final_img = self.tone_map(img, exposure, (view.width, view.height))
        return final_img.reshape(view.height, view.width, 3).astype(np.uint8)

    def trace_rays(self, view, spheres, triangles, max_bounces, num_rays):
        random_seed = np.random.randint(0, 2**16 - 1, dtype=np.int32)
        buffers = [
            self.device.buffer(data)
            for data in [
                view.to_numpy(),
                spheres,
                triangles,
                np.int32(len(spheres)),
                np.int32(len(triangles)),
                np.int32(max_bounces),
                np.int32(num_rays),
                random_seed,
            ]
        ]
        image_buffer = self.device.buffer(
            view.width * view.height * 3 * np.dtype(np.float32).itemsize
        )
        buffers.append(image_buffer)
        num_threads = view.width * view.height
        self.run_kernel("trace_rays", num_threads, buffers)
        return image_buffer
        # return np.frombuffer(image_buffer, dtype=np.float32)

    def tone_map(self, hdr_image, exposure, resolution):
        buffers = [hdr_image, self.device.buffer(np.array([exposure], dtype=np.float32))]
        # buffers = [
        #     self.device.buffer(data)
        #     for data in [
        #         hdr_image,
        #         np.array([exposure], dtype=np.float32),
        #     ]
        # ]
        width, height = resolution
        image_buffer = self.device.buffer(width * height * 3 * np.dtype(np.float32).itemsize)
        buffers.append(image_buffer)
        self.run_kernel("tone_map", width * height, buffers)
        return np.frombuffer(image_buffer, dtype=np.float32)

    def run_kernel(self, name, call_count, buffers):
        kernel_fn = self.metal_library.function(name)
        kernel_fn(call_count, *buffers)
