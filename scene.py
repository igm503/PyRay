import time
import numpy as np
import cv2
from tqdm import tqdm

from surface import Surface
from triangle import Triangle
from sphere import Sphere
from view import View
from metal import MetalTracer
from cpu import CPUTracer


class Scene:
    def __init__(self, surfaces: list[Surface]):
        self.triangles = [surface for surface in surfaces if isinstance(surface, Triangle)]
        self.spheres = [surface for surface in surfaces if isinstance(surface, Sphere)]
        self.surfaces = surfaces
        self.tracer = MetalTracer()

    def cumulative_render(
        self,
        view: View,
        num_rays: int,
        max_bounces: int,
        exposure: float = 3.0,
        device: str = "metal",
        save_dir: str | None = None,
    ):
        current_img = None
        img = np.zeros((view.height, view.width, 3))
        for i in tqdm(range(num_rays)):
            img += self.render(view, 1, max_bounces, exposure, device)
            current_img = img / (i + 1)
            current_img = np.clip(current_img, 0, 255).astype(np.uint8)
            print(f"Rendering... {i+1}/{num_rays}")
            if save_dir is not None:
                cv2.imwrite(f"{save_dir}/output_{i}.png", current_img)
            cv2.imshow("image", current_img)
            cv2.waitKey(1)
        return current_img

    def render(
        self,
        view: View,
        num_rays: int,
        max_bounces: int,
        exposure: float = 3.0,
        device: str = "metal",
    ):
        if device == "cpu":
            if not hasattr(self, "cpu_render"):
                self.cpu = CPUTracer()
            renderer = self.cpu
        elif device == "metal":
            if not hasattr(self, "metal"):
                self.metal = MetalTracer()
            renderer = self.metal
        else:
            raise ValueError(f"Unknown device {device}")
        img = renderer.render(view, self.spheres, self.triangles, num_rays, max_bounces, exposure)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
