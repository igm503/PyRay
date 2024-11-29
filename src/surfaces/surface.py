from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..types import GPUTypes


if TYPE_CHECKING:
    from ..ray import Ray, Hit


class Material:
    def __init__(
        self,
        color: list,
        reflectivity: float = 0,
        luminance: float = 0,
        transparency: float = 0,
        translucency: float = 0,
        refractive_index: float = 1,
    ):
        self.color = np.array(color)
        self.reflectivity = reflectivity
        self.luminance = luminance
        self.transparency = transparency
        self.translucency = translucency
        self.refractive_index = refractive_index

    def to_numpy(self):
        return np.array(
            (
                self.color,
                self.luminance,
                self.reflectivity,
                self.transparency,
                self.translucency,
                self.refractive_index,
            ),
            dtype=GPUTypes.material_dtype,
        )


class Surface(ABC):
    @abstractmethod
    def check_hit(self, ray: "Ray") -> "Hit | None":
        pass
