import numpy as np
from numba import njit

from ray import Ray
from surface import Surface, Hit, Material
from utils import normalize
from constants import NUMBA
from metal import MetalTracer


class Triangle(Surface):
    def __init__(self, points: np.ndarray, material: Material):
        self.points = points
        self.material = material

        self.ab = points[1] - points[0]
        self.ac = points[2] - points[0]

        self.normal = normalize(np.cross(self.ab, self.ac))

    def to_numpy(self):
        return np.array(
            (self.points[0], self.points[1], self.points[2], self.material.to_numpy()),
            dtype=MetalTracer.triangle_dtype,
        )

    def check_hit(self, ray: Ray):
        if NUMBA:
            did_hit, t = check_hit_jit(self.ab, self.ac, self.points[0], ray.origin, ray.dir)
            if not did_hit:
                return None
        else:
            epsilon = 1e-6
            # pvec = np.cross(ray.dir, self.ac)
            pvec0 = ray.dir[1] * self.ac[2] - ray.dir[2] * self.ac[1]
            pvec1 = ray.dir[2] * self.ac[0] - ray.dir[0] * self.ac[2]
            pvec2 = ray.dir[0] * self.ac[1] - ray.dir[1] * self.ac[0]

            det = self.ab[0] * pvec0 + self.ab[1] * pvec1 + self.ab[2] * pvec2

            # if det > -epsilon and det < epsilon:
            #     return None
            if det < epsilon:
                return None

            inv_det = 1.0 / det

            tvec0 = ray.origin[0] - self.points[0][0]
            tvec1 = ray.origin[1] - self.points[0][1]
            tvec2 = ray.origin[2] - self.points[0][2]

            u = (tvec0 * pvec0 + tvec1 * pvec1 + tvec2 * pvec2) * inv_det
            if u < 0.0 or u > 1.0:
                return None

            qvec0 = tvec1 * self.ab[2] - tvec2 * self.ab[1]
            qvec1 = tvec2 * self.ab[0] - tvec0 * self.ab[2]
            qvec2 = tvec0 * self.ab[1] - tvec1 * self.ab[0]

            v = (ray.dir[0] * qvec0 + ray.dir[1] * qvec1 + ray.dir[2] * qvec2) * inv_det
            if v < 0.0 or u + v > 1.0:
                return None

            t = (self.ac[0] * qvec0 + self.ac[1] * qvec1 + self.ac[2] * qvec2) * inv_det
        return Hit(
            t=t,
            normal=self.normal,
            material=self.material,
        )


@njit
def check_hit_jit(ab, ac, point0, ray_origin, ray_dir):
    epsilon = 1e-6

    pvec0 = ray_dir[1] * ac[2] - ray_dir[2] * ac[1]
    pvec1 = ray_dir[2] * ac[0] - ray_dir[0] * ac[2]
    pvec2 = ray_dir[0] * ac[1] - ray_dir[1] * ac[0]

    det = ab[0] * pvec0 + ab[1] * pvec1 + ab[2] * pvec2

    # if det > -epsilon and det < epsilon:
    #     return False, 0.0
    if det < epsilon:
        return False, 0.0

    inv_det = 1.0 / det

    tvec0 = ray_origin[0] - point0[0]
    tvec1 = ray_origin[1] - point0[1]
    tvec2 = ray_origin[2] - point0[2]

    u = (tvec0 * pvec0 + tvec1 * pvec1 + tvec2 * pvec2) * inv_det
    if u < 0.0 or u > 1.0:
        return False, 0.0

    qvec0 = tvec1 * ab[2] - tvec2 * ab[1]
    qvec1 = tvec2 * ab[0] - tvec0 * ab[2]
    qvec2 = tvec0 * ab[1] - tvec1 * ab[0]

    v = (ray_dir[0] * qvec0 + ray_dir[1] * qvec1 + ray_dir[2] * qvec2) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False, 0.0

    t = (ac[0] * qvec0 + ac[1] * qvec1 + ac[2] * qvec2) * inv_det
    return True, t
