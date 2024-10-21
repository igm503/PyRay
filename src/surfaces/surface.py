from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dataclasses import dataclass
import numpy as np

from ..types import MetalTypes


if TYPE_CHECKING:
    from ..ray import Ray, Hit


@dataclass
class Material:
    color: np.ndarray
    reflectivity: float = 0
    luminance: float = 0

    def to_numpy(self):
        return np.array(
            (self.color, self.luminance, self.reflectivity),
            dtype=MetalTypes.material_dtype,
        )


class Surface(ABC):
    @abstractmethod
    def check_hit(self, ray: "Ray") -> "Hit | None":
        pass
