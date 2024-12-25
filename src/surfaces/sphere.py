import numpy as np

from .material import Material
from ..types import GPUTypes


class Sphere:
    def __init__(self, center: list, radius: float, material: Material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def to_numpy(self):
        return np.array(
            (self.center, self.radius, self.material.to_numpy()),
            dtype=GPUTypes.sphere_dtype,
        )
