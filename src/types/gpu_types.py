from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from view import View
    from ..surfaces import Sphere, Triangle


class GPUTypes:
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
            ("transparency", np.float32),
            ("translucency", np.float32),
            ("refractive_index", np.float32),
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
            ("v0", np.float32, 3),
            ("v1", np.float32, 3),
            ("v2", np.float32, 3),
            ("material", material_dtype),
        ]
    )


def inputs_to_numpy(view: "View", spheres: list["Sphere"], triangles: list["Triangle"]):
    numpy_view = view.to_numpy()
    numpy_spheres = np.zeros(max(len(spheres), 1), dtype=GPUTypes.sphere_dtype)
    for i, sphere in enumerate(spheres):
        numpy_spheres[i] = sphere.to_numpy()
    numpy_triangles = np.zeros(max(len(triangles), 1), dtype=GPUTypes.triangle_dtype)
    for i, triangle in enumerate(triangles):
        numpy_triangles[i] = triangle.to_numpy()
    return numpy_view, numpy_spheres, numpy_triangles
