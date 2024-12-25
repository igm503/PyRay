import numpy as np

from .material import Material
from ..utils import normalize
from ..types import GPUTypes


class Triangle:
    def __init__(self, points: list, material: Material, mesh_id: int | None = None):
        self.points = np.array(points)
        self.material = material
        self.mesh_id = mesh_id if mesh_id is not None else -1

        self.v0, self.v1, self.v2 = self.points

        self.ab = self.v1 - self.v0
        self.ac = self.v2 - self.v0

        self.normal = normalize(np.cross(self.ab, self.ac))

    def to_numpy(self):
        return np.array(
            (
                self.v0,
                self.ab,
                self.ac,
                self.normal,
                self.material.to_numpy(),
                self.mesh_id,
            ),
            dtype=GPUTypes.triangle_dtype,
        )

