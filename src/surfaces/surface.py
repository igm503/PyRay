from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..types import MetalTypes


if TYPE_CHECKING:
    from ..ray import Ray, Hit


class Material:
    def __init__(self, color: list, reflectivity: float = 0, luminance: float = 0):
        self.color = np.array(color)
        self.reflectivity = reflectivity
        self.luminance = luminance

    def to_numpy(self):
        return np.array(
            (self.color, self.luminance, self.reflectivity),
            dtype=MetalTypes.material_dtype,
        )


class Surface(ABC):
    @abstractmethod
    def check_hit(self, ray: "Ray") -> "Hit | None":
        pass
