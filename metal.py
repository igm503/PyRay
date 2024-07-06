import metalcompute as mc
import numpy as np

from metal_code import metal_code


class MetalTracer:
    ray_dtype = np.dtype(
        [
            ("origin", np.float32, 3),
            ("direction", np.float32, 3),
            ("pixel", np.int32, 2),
            ("color", np.float32, 3),
            ("intensity", np.float32),
        ]
    )

    view_dtype = np.dtype(
        [
            ("origin", np.float32, 3),
            ("direction", np.float32, 3),
            ("fov", np.float32, 1),
            ("width", np.int32, 1),
            ("height", np.int32, 1),
        ]
    )

    material_dtype = np.dtype(
        [
            ("color", np.float32, 3),
            ("intensity", np.float32),
            ("reflectivity", np.float32),
        ]
    )

    sphere_dtype = np.dtype(
        [
            ("center", np.float32, 3),
            ("radius", np.float32),
            ("material", material_dtype),
        ]
    )

    triangle_dtype = np.dtype(
        [
            ("v1", np.float32, 3),
            ("v2", np.float32, 3),
            ("v3", np.float32, 3),
            ("material", material_dtype),
        ]
    )

    def __init__(self):
        self.device = mc.Device()
        self.metal_library = self.device.kernel(metal_code)

    def run_kernel(self, name, call_count, buffers):
        kernel_fn = self.metal_library.function(name)
        kernel_fn(call_count, *buffers)

    def trace_rays(
        self,
        view,
        spheres,
        triangles,
        num_spheres,
        num_triangles,
        num_bounces,
        num_rays,
    ):
        random_seed = np.random.randint(0, 2 ** 16 - 1, dtype=np.int32)
        buffers = [
            self.device.buffer(data)
            for data in [
                view.to_numpy(),
                spheres,
                triangles,
                np.array([num_spheres], dtype=np.int32),
                np.array([num_triangles], dtype=np.int32),
                np.array([num_bounces], dtype=np.int32),
                np.array([num_rays], dtype=np.int32),
                random_seed,
            ]
        ]
        image_buffer = self.device.buffer(view.width * view.height * 3 * np.dtype(np.float32).itemsize)
        buffers.append(image_buffer)
        num_threads = view.width * view.height 
        self.run_kernel("trace_rays", num_threads, buffers)
        return np.frombuffer(image_buffer, dtype=np.float32)

    def tone_map(self, hdr_image, exposure, resolution):
        buffers = [
            self.device.buffer(data)
            for data in [
                hdr_image,
                np.array([exposure], dtype=np.float32),
            ]
        ]
        width, height = resolution
        image_buffer = self.device.buffer(width * height * 3 * np.dtype(np.float32).itemsize)
        buffers.append(image_buffer)
        self.run_kernel("tone_map", width * height, buffers)
        return np.frombuffer(image_buffer, dtype=np.float32)
