from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


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
    def render(
        self,
        view: "View",
        spheres: list["Sphere"],
        triangles: list["Triangle"],
        num_rays: int,
        max_bounces: int,
        exposure: float,
    ):
        # Convert inputs to numpy arrays
        
        np_view, np_spheres, np_triangles = inputs_to_numpy(view, spheres, triangles)
        
        num_pixels = view.width * view.height
        block_size = 256
        grid_size = (num_pixels + block_size - 1) // block_size
        
        try:
            rand_states = cuda.mem_alloc(num_pixels * 48)
            image = cuda.mem_alloc(num_pixels * 3 * np.dtype(np.float32).itemsize)
            
            view_gpu = cuda.mem_alloc(np_view.nbytes)
            cuda.memcpy_htod(view_gpu, np_view)
                   
            spheres_gpu = cuda.mem_alloc(np_spheres.nbytes)
            cuda.memcpy_htod(spheres_gpu, np_spheres)
            
            triangles_gpu = cuda.mem_alloc(np_triangles.nbytes)
            cuda.memcpy_htod(triangles_gpu, np_triangles)
            
            # Initialize random states
            seed = np.random.randint(0, 2**31 - 1, dtype=np.int32)
            self.init_rand_kernel(
                rand_states,
                np.int32(seed),
                np.int32(num_pixels),  # Add size parameter
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            cuda.Context.synchronize()
            
            # Clear the image buffer
            cuda.memset_d32(image, 0, num_pixels * 3)
            
            # Trace rays
            self.trace_rays_kernel(
                view_gpu,
                spheres_gpu,
                triangles_gpu,
                np.int32(len(spheres)),
                np.int32(len(triangles)),
                np.int32(max_bounces),
                np.int32(num_rays),
                np.float32(exposure),
                rand_states,
                image,
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            
            cuda.Context.synchronize()

       
            result = np.empty(num_pixels * 3, dtype=np.float32)
            cuda.memcpy_dtoh(result, image)
            
            print(result[0], result[1], result[2])
            
            return result.reshape(view.height, view.width, 3).astype(np.uint8)
            
        finally:
            # Free GPU memory in finally block to ensure cleanup
            for gpu_array in [rand_states, image, view_gpu, spheres_gpu, triangles_gpu]:
                try:
                    gpu_array.free()
                except:
                    pass  # Ignore errors during cleanup
