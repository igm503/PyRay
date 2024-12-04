from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..types import GPUTypes


if TYPE_CHECKING:
    from ..ray import Ray, Hit


class Material:
    def __init__(
        self,
        color: list = [1, 1, 1],
        luminance: float = 0.0,
        reflectivity: float = 0.0,
        glossy: bool = False,
        gloss_refractive_index: float = 1.0,
        gloss_translucency: float = 0.0,
        transparent: bool = False,
        refractive_index: float = 1.0,
        translucency: float = 0.0,
        absorption: list = [0.01, 0.01, 0.01],
    ):
        self.color = np.array(color)
        self.reflectivity = reflectivity
        self.luminance = luminance
        self.transparent = transparent
        self.translucency = translucency
        self.refractive_index = refractive_index
        self.absorption = absorption
        self.glossy = glossy
        self.gloss_refractive_index = gloss_refractive_index
        self.gloss_translucency = gloss_translucency

    def to_numpy(self):
        return np.array(
            (
                self.color,
                self.luminance,
                self.reflectivity,
                self.transparent,
                self.refractive_index,
                self.translucency,
                self.absorption,
                self.glossy,
                self.gloss_refractive_index,
                self.gloss_translucency,
            ),
            dtype=GPUTypes.material_dtype,
        )


class Surface(ABC):
    @abstractmethod
    def check_hit(self, ray: "Ray") -> "Hit | None":
        pass
