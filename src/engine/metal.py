from typing import TYPE_CHECKING
from pathlib import Path
import ctypes

import Metal  # from pyobjc
# import metalcompute as mc
import numpy as np

from ..types import inputs_to_numpy

if TYPE_CHECKING:
    from view import View
    from ..surfaces import Sphere, Triangle


class MetalTracer:
    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()

        metal_code_path = Path(__file__).parent / "metal_code.metal"
        with open(metal_code_path, "r") as f:
            metal_code = f.read()

        # dev = mc.Device()
        # dev.kernel(metal_code)

        self.metal_library = self.device.newLibraryWithSource_options_error_(
            metal_code, None, None
        )[0]
        self.command_queue = self.device.newCommandQueue()

        self.buffer_cache = {}

    def _render_iteration(
        self,
        view: "View",
        spheres: list["Sphere"],
        triangles: list["Triangle"],
        num_rays: int,
        max_bounces: int,
        exposure: float,
        accumulate: bool,
        iteration: int,
    ):
        np_view, np_spheres, np_triangles = inputs_to_numpy(view, spheres, triangles)

        image_size = view.width * view.height * 3 * np.dtype(np.float32).itemsize
        hdr_buffer = self.get_buffer(image_size, "hdr_accum", shared=True)
        output_buffer = self.get_buffer(image_size, "output", shared=True)

        seed = np.random.randint(0, 2**16 - 1, dtype=np.int32)

        input_data = [
            np_view,
            np_spheres,
            np_triangles,
            np.int32(len(np_spheres)),
            np.int32(len(np_triangles)),
            np.int32(max_bounces),
            seed,
            np.int32(num_rays),
            np.float32(exposure),
            np.bool_(accumulate),
            np.int32(iteration),
        ]

        buffers = []
        for idx, data in enumerate(input_data):
            buffer_size = data.nbytes if hasattr(data, "nbytes") else data.itemsize
            buffer = self.get_buffer(buffer_size, idx)
            buffer_array = (ctypes.c_byte * buffer_size).from_buffer(
                buffer.contents().as_buffer(buffer_size)
            )
            buffer_array[:] = data.tobytes() if hasattr(data, "tobytes") else data
            buffers.append(buffer)

        buffers.extend([hdr_buffer, output_buffer])

        num_pixels = view.width * view.height

        self.run_kernel(num_pixels, buffers, self.trace_pipeline)

        image_size = num_pixels * 3 * np.dtype(np.float32).itemsize
        output_array = (ctypes.c_float * (num_pixels * 3)).from_buffer(
            output_buffer.contents().as_buffer(image_size)
        )
        img = np.frombuffer(output_array, dtype=np.float32)
        return img.reshape(view.height, view.width, 3).astype(np.uint8)

    def render(
        self,
        view: "View",
        spheres: list["Sphere"],
        triangles: list["Triangle"],
        num_rays: int,
        max_bounces: int,
        exposure: float,
    ):
        return self._render_iteration(
            view,
            spheres,
            triangles,
            num_rays,
            max_bounces,
            exposure,
            False,
            0,
        )

    def cumulative_render(
        self,
        view: "View",
        spheres: list["Sphere"],
        triangles: list["Triangle"],
        num_rays: int,
        max_bounces: int,
        exposure: float,
        num_iterations: int,
    ):
        for iteration in range(num_iterations):
            yield self._render_iteration(
                view,
                spheres,
                triangles,
                num_rays,
                max_bounces,
                exposure,
                True,
                iteration,
            )

    def get_buffer(self, size, idx, shared=False):
        if idx in self.buffer_cache:
            if size in self.buffer_cache[idx]:
                return self.buffer_cache[idx][size]
            del self.buffer_cache[idx]
        mode = Metal.MTLResourceStorageModePrivate
        if shared:
            mode = Metal.MTLResourceStorageModeShared
        buffer = self.device.newBufferWithLength_options_(size, mode)
        self.buffer_cache[idx] = {size: buffer}
        return buffer

    def run_kernel(self, num_threads, buffers, pipeline, wait=True):
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        compute_encoder.setComputePipelineState_(pipeline)

        for idx, buffer in enumerate(buffers):
            compute_encoder.setBuffer_offset_atIndex_(buffer, 0, idx)

        max_threads = pipeline.maxTotalThreadsPerThreadgroup()
        threads_per_threadgroup = Metal.MTLSizeMake(max_threads, 1, 1)
        threads = Metal.MTLSizeMake(num_threads, 1, 1)

        compute_encoder.dispatchThreads_threadsPerThreadgroup_(threads, threads_per_threadgroup)
        compute_encoder.endEncoding()

        command_buffer.commit()
        if wait:
            command_buffer.waitUntilCompleted()  # faster without wait, but screen tearing on movement
