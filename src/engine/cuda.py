from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from ..types import inputs_to_numpy

if TYPE_CHECKING:
    from view import View
    from ..surfaces import Sphere, Triangle


class CudaTracer:
    def __init__(self):
        cuda_code_path = Path(__file__).parent / "cuda_code.cu"
        with open(cuda_code_path, "r") as f:
            cuda_code = f.read()
        self.module = SourceModule(cuda_code, arch="sm_86", no_extern_c=True)
        self.trace_rays_kernel = self.module.get_function("trace_rays")
        self.init_rand_kernel = self.module.get_function("init_rand_state")
        self.rand_states = None
        self.num_pixels = None

        self.buffer_cache = {}

    def render(
        self,
        view: "View",
        spheres: list["Sphere"],
        triangles: list["Triangle"],
        num_rays: int,
        max_bounces: int,
        exposure: float,
    ):
        num_pixels = view.width * view.height
        block_size = 256
        grid_size = (num_pixels + block_size - 1) // block_size
        np_view, np_spheres, np_triangles = inputs_to_numpy(view, spheres, triangles)

        image_size = num_pixels * 3 * np.dtype(np.float32).itemsize
        image_buffer = self.get_buffer(image_size, "image")
        view_buffer = self.get_buffer(np_view.nbytes, "view")
        spheres_buffer = self.get_buffer(np_spheres.nbytes, "spheres")
        triangles_buffer = self.get_buffer(np_triangles.nbytes, "triangles")

        random_states = self.get_random_states(num_pixels, grid_size, block_size)

        cuda.memcpy_htod(view_buffer, np_view)
        cuda.memcpy_htod(spheres_buffer, np_spheres)
        cuda.memcpy_htod(triangles_buffer, np_triangles)

        self.trace_rays_kernel(
            view_buffer,
            spheres_buffer,
            triangles_buffer,
            np.int32(len(spheres)),
            np.int32(len(triangles)),
            np.int32(max_bounces),
            np.int32(num_rays),
            np.float32(exposure),
            random_states,
            image_buffer,
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        result = np.empty(num_pixels * 3, dtype=np.float32)
        cuda.memcpy_dtoh(result, image_buffer)
        return result.reshape(view.height, view.width, 3).astype(np.uint8)

    def get_buffer(self, size, name):
        if name in self.buffer_cache:
            if size in self.buffer_cache[name]:
                return self.buffer_cache[name][size]
            del self.buffer_cache[name]
        buffer = cuda.mem_alloc(size)
        self.buffer_cache[name] = {size: buffer}
        return buffer

    def get_random_states(self, num_pixels, grid_size, block_size):
        if self.rand_states is None or self.num_pixels != num_pixels:
            if self.rand_states is not None:
                self.rand_states.free()
            self.rand_states = cuda.mem_alloc(num_pixels * 48)  # curand state is 48 bytes
            seed = np.random.randint(0, 2**31 - 1, dtype=np.int32)
            self.init_rand_kernel(
                self.rand_states,
                np.int32(seed),
                np.int32(num_pixels),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
            )
            self.num_pixels = num_pixels
        return self.rand_states
