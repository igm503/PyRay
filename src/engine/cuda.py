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

    def render_iteration(
        self,
        view: "View",
        spheres: list["Sphere"],
        triangles: list["Triangle"],
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

        num_pixels = view.width * view.height
        block_size = 256
        grid_size = (num_pixels + block_size - 1) // block_size

        # copied each render
        view_buffer = self.get_buffer(np_view.nbytes, "view")
        cuda.memcpy_htod(view_buffer, np_view)

        num_surrounding = len(surrounding_spheres)
        surrounding_buffer = self.get_buffer(
            np_surrounding.nbytes, "surrounding_spheres", cache_data=np_surrounding
        )
        cuda.memcpy_htod(surrounding_buffer, np_surrounding)

        random_states = self.get_random_states(num_pixels, grid_size, block_size)

        # not copied each render
        spheres_buffer = self.get_buffer(np_spheres.nbytes, "spheres", cache_data=np_spheres)
        triangles_buffer = self.get_buffer(
            np_triangles.nbytes, "triangles", cache_data=np_triangles
        )
        background_color_buffer = self.get_buffer(
            np_background_color.nbytes,
            "background_color",
            cache_data=np_background_color,
        )

        image_size = num_pixels * 3 * np.dtype(np.float32).itemsize
        accumulation_buffer = self.get_buffer(image_size, "accumulation")
        out_buffer = self.get_buffer(image_size, "out")

        self.trace_rays_kernel(
            view_buffer,
            random_states,
            spheres_buffer,
            triangles_buffer,
            surrounding_buffer,
            np.int32(len(spheres)),
            np.int32(len(triangles)),
            np.int32(num_surrounding),
            np.int32(max_bounces),
            np.int32(num_rays),
            background_color_buffer,
            np.float32(background_luminance),
            np.float32(exposure),
            np.int32(accumulate),
            np.int32(iteration),
            accumulation_buffer,
            out_buffer,
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        img = np.empty(num_pixels * 3, dtype=np.float32)
        cuda.memcpy_dtoh(img, out_buffer)
        return img.reshape(view.height, view.width, 3).astype(np.uint8)

    def get_buffer(self, size, name, cache_data=None):
        if name in self.buffer_cache:
            if size in self.buffer_cache[name]:
                return self.buffer_cache[name][size]
            else:
                del self.buffer_cache[name]
        return self.create_buffer(size, name, cache_data=cache_data)

    def create_buffer(self, size, name, cache_data=None):
        buffer = cuda.mem_alloc(size)
        self.buffer_cache[name] = {size: buffer}
        if cache_data is not None:
            cuda.memcpy_htod(buffer, cache_data)
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

    def clear_cache(self):
        self.buffer_cache = {}
