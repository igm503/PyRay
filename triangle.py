import numpy as np
from numba import njit

from ray import Ray
from surface import Surface


class Triangle(Surface):
    def __init__(self, points: np.ndarray, color: np.ndarray, material: str):
        self.points = points
        self.color = color
        self.material = material

        self.x_min = np.min(points[:, 0])
        self.y_min = np.min(points[:, 1])
        self.x_max = np.max(points[:, 0])
        self.y_max = np.max(points[:, 1])
        self.z_min = np.min(points[:, 2])
        self.z_max = np.max(points[:, 2])

        self.ab = points[1] - points[0]
        self.ac = points[2] - points[0]

        self.normal = np.cross(self.ab, self.ac)
        self.normal = self.normal / np.linalg.norm(self.normal)

    def check_hit(self, ray: Ray):
        # return check_hit_jit(self.ab, self.ac, self.points[0], ray.origin, ray.dir, self.color, self.normal)
        # pvec = np.cross(ray.dir, self.ac)
        pvec = np.array(
            [
                ray.dir[1] * self.ac[2] - ray.dir[2] * self.ac[1],
                ray.dir[2] * self.ac[0] - ray.dir[0] * self.ac[2],
                ray.dir[0] * self.ac[1] - ray.dir[1] * self.ac[0],
            ]
        )
        # det = self.ab.dot(pvec)
        det = self.ab[0] * pvec[0] + self.ab[1] * pvec[1] + self.ab[2] * pvec[2]
        if np.abs(det) < 1e-6:
            return False, None, None, None
        inv_det = 1 / det
        tvec = ray.origin - self.points[0]
        # u = tvec.dot(pvec) / det
        u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det
        if u < 0 or u > 1:
            return False, None, None, None
        # qvec = np.cross(tvec, self.ab)
        qvec = np.array(
            [
                tvec[1] * self.ab[2] - tvec[2] * self.ab[1],
                tvec[2] * self.ab[0] - tvec[0] * self.ab[2],
                tvec[0] * self.ab[1] - tvec[1] * self.ab[0],
            ]
        )
        # v = ray.dir.dot(qvec) * inv_det
        v = (ray.dir[0] * qvec[0] + ray.dir[1] * qvec[1] + ray.dir[2] * qvec[2]) * inv_det
        if v < 0 or u + v > 1:
            return False, None, None, None
        # t = self.ac.dot(qvec) * inv_det
        t = (self.ac[0] * qvec[0] + self.ac[1] * qvec[1] + self.ac[2] * qvec[2]) * inv_det
        return True, self.color, self.normal, t


@njit
def check_hit_jit(ab, ac, p0, ray_origin, ray_dir, color, normal):
    pvec = np.array(
        [
            ray_dir[1] * ac[2] - ray_dir[2] * ac[1],
            ray_dir[2] * ac[0] - ray_dir[0] * ac[2],
            ray_dir[0] * ac[1] - ray_dir[1] * ac[0],
        ]
    )

    det = ab[0] * pvec[-1] + ab[1] * pvec[1] + ab[2] * pvec[2]
    if np.abs(det) < 1e-6:
        return False, None, None, None

    inv_det = 1.0 / det

    tvec = ray_origin - p0

    u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det
    if u < 0 or u > 1:
        return False, None, None, None

    qvec = np.array(
        [
            tvec[1] * ab[2] - tvec[2] * ab[1],
            tvec[2] * ab[0] - tvec[0] * ab[2],
            tvec[0] * ab[1] - tvec[1] * ab[0],
        ]
    )

    v = (ray_dir[0] * qvec[0] + ray_dir[1] * qvec[1] + ray_dir[2] * qvec[2]) * inv_det
    if v < 0 or u + v > 1:
        return False, None, None, None

    t = (ac[0] * qvec[0] + ac[1] * qvec[1] + ac[2] * qvec[2]) * inv_det

    return True, color, normal, t
