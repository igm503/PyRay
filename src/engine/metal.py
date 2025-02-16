from pathlib import Path
import ctypes

import Metal  # from pyobjc
import metalcompute as mc
import numpy as np

from ..types import inputs_to_numpy
from ..view import View
from ..surfaces import Sphere, Triangle


class MetalTracer:
    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()

        metal_code_path = Path(__file__).parent / "metal_code.metal"
        with open(metal_code_path, "r") as f:
            metal_code = f.read()

        dev = mc.Device()
        dev.kernel(metal_code)

        self.metal_library = self.device.newLibraryWithSource_options_error_(
            metal_code, None, None
        )[0]
        self.command_queue = self.device.newCommandQueue()

        trace_function = self.metal_library.newFunctionWithName_("trace_rays")
        self.trace_pipeline = self.device.newComputePipelineStateWithFunction_error_(
            trace_function, None
        )[0]

        self.buffer_cache = {}

    def render_iteration(
        self,
        view: View,
        spheres: list[Sphere],
        triangles: list[Triangle],
        surrounding_spheres: list[int],
        num_rays: int,
        max_bounces: int,
        background_color: tuple[float, float, float],
        background_luminance: float,
        exposure: float,
        accumulate: bool,
        iteration: int = 0,
    ):
        np_view, np_spheres, np_triangles, np_surrounding, np_background_color = inputs_to_numpy(
            view, spheres, triangles, surrounding_spheres, background_color
        )

        seed = np.random.randint(0, 2**16 - 1, dtype=np.int32)

        num_surrounding = len(surrounding_spheres)

        input_data = [
            (np_view, "view"),
            (seed, "seed"),
            (np_spheres, "spheres"),
            (np_triangles, "triangles"),
            (np_surrounding, "surrounding_spheres"),
            (np.int32(len(np_spheres)), "num_spheres"),
            (np.int32(len(np_triangles)), "num_triangles"),
            (np.int32(num_surrounding), "num_surrounding_spheres"),
            (np.int32(max_bounces), "max_bounces"),
            (np.int32(num_rays), "num_rays"),
            (np_background_color, "background_color"),
            (np.float32(background_luminance), "background_luminance"),
            (np.float32(exposure), "exposure"),
            (np.bool_(accumulate), "accumulate"),
            (np.int32(iteration), "iteration"),
        ]

        buffers = []
        for data, name in input_data:
            buffer_size = data.nbytes if hasattr(data, "nbytes") else data.itemsize
            buffer = self.get_buffer(buffer_size, name, cache_data=data)
            if name in [
                "view",
                "seed",
                "surrounding_spheres",
                "num_surrounding_spheres",
            ]:
                self.load_buffer(buffer, data, buffer_size)
            elif name in "iteration" and accumulate:
                self.load_buffer(buffer, data, buffer_size)
            buffers.append(buffer)

        num_pixels = view.width * view.height
        image_size = num_pixels * 3 * np.dtype(np.float32).itemsize
        accumulation_buffer = self.get_buffer(image_size, "accumulation", shared=True)
        out_buffer = self.get_buffer(image_size, "out", shared=True)

        buffers.extend([accumulation_buffer, out_buffer])

        self.run_kernel(num_pixels, buffers, self.trace_pipeline)

        output_array = (ctypes.c_float * (num_pixels * 3)).from_buffer(
            out_buffer.contents().as_buffer(image_size)
        )
        img = np.frombuffer(output_array, dtype=np.float32)
        return img.reshape(view.height, view.width, 3).astype(np.uint8)

    def get_buffer(self, size, name, shared=False, cache_data=None):
        if name in self.buffer_cache:
            if size in self.buffer_cache[name]:
                return self.buffer_cache[name][size]
            else:
                del self.buffer_cache[name]
        return self.create_buffer(size, name, shared, cache_data)

    def create_buffer(self, size, name, shared=False, cache_data=None):
        mode = Metal.MTLResourceStorageModePrivate
        if shared:
            mode = Metal.MTLResourceStorageModeShared
        buffer = self.device.newBufferWithLength_options_(size, mode)
        self.buffer_cache[name] = {size: buffer}
        if cache_data is not None:
            self.load_buffer(buffer, cache_data, size)
        return buffer

    def load_buffer(self, buffer, data, size):
        buffer_array = (ctypes.c_byte * size).from_buffer(buffer.contents().as_buffer(size))
        buffer_array[:] = data.tobytes() if hasattr(data, "tobytes") else data

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

    def clear_cache(self):
        self.buffer_cache = {}
