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
        self.image_buffer = None
        self.image_size = 0
        self.view_buffer = None
        self.view_size = 0
        self.spheres_buffer = None
        self.spheres_size = 0
        self.triangles_buffer = None
        self.triangles_size = 0

    def get_or_alloc_buffer(self, old_buffer, old_size, new_size):
        if old_buffer is None or old_size < new_size:
            if old_buffer is not None:
                old_buffer.free()
            return cuda.mem_alloc(new_size), new_size
        return old_buffer, old_size

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
        num_pixels = view.width * view.height
        block_size = 256
        grid_size = (num_pixels + block_size - 1) // block_size

        image_size = num_pixels * 3 * np.dtype(np.float32).itemsize
        self.image_buffer, self.image_size = self.get_or_alloc_buffer(
            self.image_buffer, self.image_size, image_size
        )
        self.view_buffer, self.view_size = self.get_or_alloc_buffer(
            self.view_buffer, self.view_size, np_view.nbytes
        )
        self.spheres_buffer, self.spheres_size = self.get_or_alloc_buffer(
            self.spheres_buffer, self.spheres_size, np_spheres.nbytes
        )
        self.triangles_buffer, self.triangles_size = self.get_or_alloc_buffer(
            self.triangles_buffer, self.triangles_size, np_triangles.nbytes
        )
        random_states = self.get_random_states(num_pixels, grid_size, block_size)

        cuda.memcpy_htod(self.view_buffer, np_view)
        cuda.memcpy_htod(self.spheres_buffer, np_spheres)
        cuda.memcpy_htod(self.triangles_buffer, np_triangles)

        self.trace_rays_kernel(
            self.view_buffer,
            self.spheres_buffer,
            self.triangles_buffer,
            np.int32(len(spheres)),
            np.int32(len(triangles)),
            np.int32(max_bounces),
            np.int32(num_rays),
            np.float32(exposure),
            random_states,
            self.image_buffer,
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        result = np.empty(num_pixels * 3, dtype=np.float32)
        cuda.memcpy_dtoh(result, self.image_buffer)
        return result.reshape(view.height, view.width, 3).astype(np.uint8)

    def get_random_states(self, num_pixels, grid_size, block_size):
        if self.rand_states is None or self.num_pixels != num_pixels:
            if self.rand_states is not None:
                self.rand_states.free()
            self.rand_states = cuda.mem_alloc(num_pixels * 48)
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

    def __del__(self):
        if self.rand_states is not None:
            self.rand_states.free()
        if self.image_buffer is not None:
            self.image_buffer.free()
        if self.view_buffer is not None:
            self.view_buffer.free()
        if self.spheres_buffer is not None:
            self.spheres_buffer.free()
        if self.triangles_buffer is not None:
            self.triangles_buffer.free()
