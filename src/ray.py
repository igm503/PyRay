from dataclasses import dataclass

import numpy as np

from .surfaces import Material


@dataclass
class Hit:
    t: float
    internal: bool
    normal: np.ndarray
    material: Material
    mesh_id: int


@dataclass
class Ray:
    origin: np.ndarray
    dir: np.ndarray
    color: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    intensity: float = 0.0
