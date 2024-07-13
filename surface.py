from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dataclasses import dataclass
import numpy as np

from metal import MetalTracer

if TYPE_CHECKING:
    from ray import Ray


@dataclass
class Material:
    color: np.ndarray
    reflectivity: float = 0
    luminance: float = 0

    def to_numpy(self):
        return np.array(
            (self.color, self.luminance, self.reflectivity),
            dtype=MetalTracer.material_dtype,
        )


@dataclass
class Hit:
    t: float
    normal: np.ndarray
    material: Material

class Surface(ABC):
    @abstractmethod
    def check_hit(self, ray: "Ray") -> Hit | None:
        pass
