import numpy as np

class MetalTypes:
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
            ("top_left_dir", np.float32, 3),
            ("right_dir", np.float32, 3),
            ("down_dir", np.float32, 3),
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
