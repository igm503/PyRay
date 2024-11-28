from typing import TYPE_CHECKING
from pathlib import Path
import ctypes

import Metal  # from pyobjc
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

        self.metal_library = self.device.newLibraryWithSource_options_error_(
            metal_code, None, None
        )[0]
        self.command_queue = self.device.newCommandQueue()

        kernel_function = self.metal_library.newFunctionWithName_("trace_rays")
        self.pipeline_state = self.device.newComputePipelineStateWithFunction_error_(
            kernel_function, None
        )[0]

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
        img = self.trace_rays(view, np_spheres, np_triangles, max_bounces, num_rays, exposure)
        return img.reshape(view.height, view.width, 3).astype(np.uint8)

    def trace_rays(self, view, spheres, triangles, max_bounces, num_rays, exposure):
        random_seed = np.random.randint(0, 2**16 - 1, dtype=np.int32)

        input_data = [
            view.to_numpy(),
            spheres,
            triangles,
            np.int32(len(spheres)),
            np.int32(len(triangles)),
            np.int32(max_bounces),
            np.int32(num_rays),
            random_seed,
            np.float32(exposure),
        ]

        buffers = []
        for data in input_data:
            buffer_size = data.nbytes
            metal_buffer = self.device.newBufferWithLength_options_(
                buffer_size, Metal.MTLResourceStorageModeShared
            )
            buffer_array = (ctypes.c_byte * buffer_size).from_buffer(
                metal_buffer.contents().as_buffer(buffer_size)
            )
            buffer_array[:] = data.tobytes()
            buffers.append(metal_buffer)

        image_size = view.width * view.height * 3 * np.dtype(np.float32).itemsize
        image_buffer = self.device.newBufferWithLength_options_(
            image_size, Metal.MTLResourceStorageModeShared
        )
        buffers.append(image_buffer)

        self.run_kernel(view.width * view.height, buffers)

        output_array = (ctypes.c_float * (view.width * view.height * 3)).from_buffer(
            image_buffer.contents().as_buffer(image_size)
        )
        return np.frombuffer(output_array, dtype=np.float32)

    def run_kernel(self, num_threads, buffers):
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        compute_encoder.setComputePipelineState_(self.pipeline_state)

        for idx, buffer in enumerate(buffers):
            compute_encoder.setBuffer_offset_atIndex_(buffer, 0, idx)

        threads_per_threadgroup = min(self.pipeline_state.maxTotalThreadsPerThreadgroup(), 1000000)

        threads = Metal.MTLSizeMake(num_threads, 1, 1)
        threadgroups = Metal.MTLSizeMake(threads_per_threadgroup, 1, 1)

        compute_encoder.dispatchThreads_threadsPerThreadgroup_(threads, threadgroups)
        compute_encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()
