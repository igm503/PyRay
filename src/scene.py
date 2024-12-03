import os

import numpy as np
import cv2
from tqdm import tqdm

from .surfaces import Surface, Triangle, Sphere
from .view import View
from .engine import get_engine


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
        device: str = "cpu",
    ):
        surrounding_media = self.get_surrounding_media(view, self.spheres)
        if surrounding_media:
            print(len(surrounding_media))
        engine = self.engine(device)
        img = engine.render(
            view,
            self.spheres,
            self.triangles,
            surrounding_media,
            num_rays,
            max_bounces,
            exposure,
        )
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def cumulative_render(
        self,
        view: View,
        num_rays: int,
        max_bounces: int,
        exposure: float = 3.0,
        device: str = "cpu",
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

        surrounding_media = self.get_surrounding_media(view, self.spheres)
        engine = self.engine(device)
        print(f"Rendering {num_rays} rays. Will save to {save_dir}")
        rays_per_frame = 100
        num_iterations = num_rays // rays_per_frame
        save_interval = num_iterations // 10
        for i, img in tqdm(
            enumerate(
                engine.cumulative_render(
                    view,
                    self.spheres,
                    self.triangles,
                    surrounding_media,
                    rays_per_frame,
                    max_bounces,
                    exposure,
                    num_iterations,
                )
            ),
            total=num_iterations,
        ):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", img)
            cv2.waitKey(1)
            if save_dir is not None and i % save_interval == 0:
                cv2.imwrite(f"{save_dir}/{i * rays_per_frame}_rays.png", img)
        if save_dir is not None:
            cv2.imwrite(f"{save_dir}/output.png", img)
        print(f"Renders are saved to {save_dir}")

    def engine(self, device: str):
        if not hasattr(self, device):
            setattr(self, device, get_engine(device))
        return getattr(self, device)

    def get_surrounding_media(self, view: View, spheres: list[Sphere]):
        surrounding_spheres = {}
        for i, sphere in enumerate(spheres):
            v_center_offset = view.origin - sphere.center
            center_offset = np.linalg.norm(v_center_offset)
            if center_offset < sphere.radius:
                surrounding_spheres[sphere.radius - center_offset] = i

        return [surrounding_spheres[key] for key in sorted(surrounding_spheres)]
