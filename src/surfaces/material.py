import numpy as np

from ..types import GPUTypes


class Material:
    def __init__(
        self,
        color: list[float] = [1.0, 1.0, 1.0],
        luminance: float = 0.0,
        reflectivity: float = 0.0,
        glossy: bool = False,
        gloss_refractive_index: float = 1.0,
        gloss_translucency: float = 0.0,
        transparent: bool = False,
        refractive_index: float = 1.0,
        translucency: float = 0.0,
        absorption: list[float] = [0.01, 0.01, 0.01],
    ):
        self.color = np.array(color)
        assert reflectivity >= 0 and reflectivity <= 1
        self.reflectivity = reflectivity
        assert luminance >= 0
        self.luminance = luminance
        assert transparent is True or transparent is False
        self.transparent = transparent
        assert translucency >= 0 and translucency <= 1
        self.translucency = translucency
        assert refractive_index >= 0
        self.refractive_index = refractive_index
        assert all([i >= 0 for i in absorption])
        self.absorption = absorption
        assert glossy is True or glossy is False
        self.glossy = glossy
        assert gloss_refractive_index >= 0
        self.gloss_refractive_index = gloss_refractive_index
        assert gloss_translucency >= 0
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
