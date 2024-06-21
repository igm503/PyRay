import numpy as np
from numba import njit

from ray import Ray
from surface import Surface, Hit

NUMBA = True

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
        if NUMBA:
            hit, u, v, t = check_hit_jit(self.ab, self.ac, self.points[0], ray.origin, ray.dir)
            if hit:
                return Hit(color=self.color, normal=self.normal, t=t, material=self.material)
            return None
        else:
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
                return None
            inv_det = 1 / det
            tvec = ray.origin - self.points[0]
            # u = tvec.dot(pvec) / det
            u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det
            if u < 0 or u > 1:
                return None
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
                return None
            # t = self.ac.dot(qvec) * inv_det
            t = (self.ac[0] * qvec[0] + self.ac[1] * qvec[1] + self.ac[2] * qvec[2]) * inv_det
            return Hit(color=self.color, normal=self.normal, t=t, material=self.material)

@njit
def check_hit_jit(ab, ac, point0, ray_origin, ray_dir):
    epsilon = 1e-6
    
    # Calculate pvec directly
    pvec0 = ray_dir[1] * ac[2] - ray_dir[2] * ac[1]
    pvec1 = ray_dir[2] * ac[0] - ray_dir[0] * ac[2]
    pvec2 = ray_dir[0] * ac[1] - ray_dir[1] * ac[0]
    
    det = ab[0] * pvec0 + ab[1] * pvec1 + ab[2] * pvec2
    
    if det > -epsilon and det < epsilon:
        return False, 0.0, 0.0, 0.0
    
    inv_det = 1.0 / det
    
    tvec0 = ray_origin[0] - point0[0]
    tvec1 = ray_origin[1] - point0[1]
    tvec2 = ray_origin[2] - point0[2]
    
    u = (tvec0 * pvec0 + tvec1 * pvec1 + tvec2 * pvec2) * inv_det
    if u < 0.0 or u > 1.0:
        return False, 0.0, 0.0, 0.0
    
    # Calculate qvec directly
    qvec0 = tvec1 * ab[2] - tvec2 * ab[1]
    qvec1 = tvec2 * ab[0] - tvec0 * ab[2]
    qvec2 = tvec0 * ab[1] - tvec1 * ab[0]
    
    v = (ray_dir[0] * qvec0 + ray_dir[1] * qvec1 + ray_dir[2] * qvec2) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False, 0.0, 0.0, 0.0
    
    t = (ac[0] * qvec0 + ac[1] * qvec1 + ac[2] * qvec2) * inv_det
    return True, u, v, t
