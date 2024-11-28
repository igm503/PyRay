import os

import numpy as np
import cv2
from tqdm import tqdm

from .surfaces import Surface, Triangle, Sphere
from .view import View
from .engine import MetalTracer, CPUTracer


class Scene:
    def __init__(self, surfaces: list[Surface]):
        self.triangles = [surface for surface in surfaces if isinstance(surface, Triangle)]
        self.spheres = [surface for surface in surfaces if isinstance(surface, Sphere)]

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

    def cumulative_render(
        self,
        view: View,
        num_rays: int,
        max_bounces: int,
        exposure: float = 3.0,
        device: str = "metal",
        save_dir: str | None = None,
    ):
        if save_dir is not None:
            sub_dir = "1"
            path = os.path.join(save_dir, sub_dir)
            while os.path.exists(path):
                sub_dir = str(int(sub_dir) + 1)
                path = os.path.join(save_dir, sub_dir)
            os.makedirs(path)
            save_dir = path

        current_img = None
        img = np.zeros((view.height, view.width, 3))
        print(f"Rendering {num_rays} rays. Will save to {save_dir}")
        for i in tqdm(range(num_rays)):
            img += self.render(view, 1, max_bounces, exposure, device)
            current_img = img / (i + 1)
            current_img = np.clip(current_img, 0, 255).astype(np.uint8)
            if save_dir is not None:
                cv2.imwrite(f"{save_dir}/output_{i}.png", current_img)
            cv2.imshow("image", current_img)
            cv2.waitKey(1)
        print(f"Renders are saved to {save_dir}")
        return current_img
