from typing import TYPE_CHECKING
from pathlib import Path
import ctypes

import Metal  # from pyobjc
import metalcompute as mc
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

        dev = mc.Device()
        dev.kernel(metal_code)

        self.metal_library = self.device.newLibraryWithSource_options_error_(
            metal_code, None, None
        )[0]
        self.command_queue = self.device.newCommandQueue()

        kernel_function = self.metal_library.newFunctionWithName_("trace_rays")
        self.pipeline_state = self.device.newComputePipelineStateWithFunction_error_(
            kernel_function, None
        )[0]

        self.buffer_cache = {}
        self.first = 0

    def render(
        self,
        view: "View",
        spheres: list["Sphere"],
        triangles: list["Triangle"],
        num_rays: int,
        max_bounces: int,
        exposure: float,
    ):
        np_view, np_spheres, np_triangles = inputs_to_numpy(view, spheres, triangles)

        seed = np.random.randint(0, 2**16 - 1, dtype=np.int32)

        input_data = [
            np_view,
            np_spheres,
            np_triangles,
            np.int32(len(spheres)),
            np.int32(len(triangles)),
            np.int32(max_bounces),
            np.int32(num_rays),
            seed,
            np.float32(exposure),
        ]

        buffers = []
        for idx, data in enumerate(input_data):
            buffer_size = data.nbytes
            buffer = self.get_buffer(buffer_size, idx)
            buffer_array = (ctypes.c_byte * buffer_size).from_buffer(
                buffer.contents().as_buffer(buffer_size)
            )
            buffer_array[:] = data.tobytes()
            buffers.append(buffer)

        image_size = view.width * view.height * 3 * np.dtype(np.float32).itemsize
        image_buffer = self.get_buffer(image_size, len(buffers), shared=True)
        buffers.append(image_buffer)

        self.run_kernel(view.width * view.height, buffers)

        output_array = (ctypes.c_float * (view.width * view.height * 3)).from_buffer(
            image_buffer.contents().as_buffer(image_size)
        )
        img = np.frombuffer(output_array, dtype=np.float32)
        return img.reshape(view.height, view.width, 3).astype(np.uint8)

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

    def run_kernel(self, num_threads, buffers):
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        compute_encoder.setComputePipelineState_(self.pipeline_state)

        for idx, buffer in enumerate(buffers):
            compute_encoder.setBuffer_offset_atIndex_(buffer, 0, idx)

        max_threads = self.pipeline_state.maxTotalThreadsPerThreadgroup()
        threads_per_threadgroup = Metal.MTLSizeMake(max_threads, 1, 1)
        threads = Metal.MTLSizeMake(num_threads, 1, 1)

        compute_encoder.dispatchThreads_threadsPerThreadgroup_(threads, threads_per_threadgroup)
        compute_encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()  # faster without wait, but screen tearing on movement
