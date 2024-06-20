from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from ray import Ray


class Surface(ABC):

    @abstractmethod
    def check_hit(self):
        pass

class Material:
    SPECULAR = "specular"
    DIFFUSE = "diffuse"


@dataclass
class Hit:
    color: np.ndarray | None = None
    normal: np.ndarray | None = None
    t: float | None = None
    material: str | None = None
