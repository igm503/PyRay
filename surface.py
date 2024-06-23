from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from ray import Ray


class Material:
    SPECULAR = "specular"
    DIFFUSE = "diffuse"


@dataclass
class Hit:
    t: float
    normal: np.ndarray
    color: np.ndarray = np.array([0, 0, 0])
    material: str = Material.DIFFUSE
    luminance: float = 0


class Surface(ABC):
    @abstractmethod
    def check_hit(self, ray: "Ray") -> Hit | None:
        pass
