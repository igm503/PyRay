from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import cupy as cp

from ..types import inputs_to_numpy

if TYPE_CHECKING:
    from view import View
    from ..surfaces import Sphere, Triangle


class CudaTracer:
    def __init__(self):
        cuda_code_path = Path(__file__).parent / "cuda_code.cu"
        with open(cuda_code_path, "r") as f:
            cuda_code = f.read()

        self.module = cp.RawModule(code=cuda_code)
        self.trace_rays_kernel = self.module.get_function("trace_rays")
        self.tone_map_kernel = self.module.get_function("tone_map")
        self.init_rand_kernel = self.module.get_function("init_rand_state")

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

        rand_states = cp.empty((num_pixels,), dtype=cp.uint64)
        seed = np.random.randint(0, 2**31 - 1)
        self.init_rand_kernel((grid_size,), (block_size,), (rand_states, seed))

        image = cp.zeros((num_pixels, 3), dtype=cp.float32)

        self.trace_rays_kernel(
            (grid_size,),
            (block_size,),
            (
                cp.asarray(np_view),
                cp.asarray(np_spheres),
                cp.asarray(np_triangles),
                np.int32(len(spheres)),
                np.int32(len(triangles)),
                np.int32(max_bounces),
                np.int32(num_rays),
                rand_states,
                image,
            ),
        )

        # Tone mapping
        final_image = cp.empty_like(image)
        self.tone_map_kernel(
            (grid_size,),
            (block_size,),
            (image, np.float32(exposure), final_image, np.int32(num_pixels)),
        )

        # Convert back to numpy and reshape
        result = cp.asnumpy(final_image)
        return result.reshape(view.height, view.width, 3).astype(np.uint8)
