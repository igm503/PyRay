import time
import math

import cv2
import numpy as np

from scene import Scene
from triangle import Triangle
from sphere import Sphere
from view import View

triangle1 = Triangle(
    points=np.array([[-150.0, -150.0, 0], [150, 150.0, -0], [-150.0, 150, -0]]),
    color=np.array([0, 0.5, 0]),
    material="diffuse",
)
triangle2 = Triangle(
    points=np.array([[-150.0, -150.0, 0], [150, -150.0, 0], [150, 150.0, -0]]),
    color=np.array([0, 0.5, 0]),
    material="diffuse",
)

sun = Sphere(
    center=np.array([-30.0, 100.0, 11.0]),
    radius=10,
    color=np.array([1, 1, 1]),
    material="diffuse",
    luminance=0.0,
)
red_sun = Sphere(
    center=np.array([50.0, 70.0, 20.0]),
    radius=10,
    color=np.array([0.7, 0.7, 1]),
    material="diffuse",
    luminance=1.0,
)
sphere1 = Sphere(
    center=np.array([0.0, 5.0, 2.0]),
    radius=1,
    color=np.array([0.5, 0, 0]),
    material="diffuse",
    luminance=0.0,
)
sphere2 = Sphere(
    center=np.array([2.0, 7.0, 6.0]),
    radius=2,
    color=np.array([0.5, 0, 0]),
    material="specular",
    luminance=0.0,
)

sphere3 = Sphere(
    center=np.array([3.0, 7, 2.0]),
    radius=2,
    color=np.array([0.5, 0, 0]),
    material="specular",
    luminance=0.0,
)

big_ball = Sphere(
    center=np.array([3.0, 7.0, -300]),
    radius=300,
    color=np.array([0.5, 0, 0.7]),
    material="diffuse",
    luminance=0.0,
)

back_ball = Sphere(
    center=np.array([-4.0, -6.0, 5]),
    radius=5,
    color=np.array([0, 0.7, 0.1]),
    material="diffuse",
)

view = View(
    origin=np.array([0, -1.0, 2.0]),
    dir=np.array([0, 0.2, 0]),
    width=3840,
    height=2160,
    # width=180,
    # height=90,
    fov=100,
)

scene = Scene(
    [back_ball, big_ball, red_sun, sun, sphere1, sphere2, sphere3]
)  # , triangle1, triangle2])

img = scene.static_render(view, num_rays=200, max_bounces=30, save_dir="big_ball")
last_time = time.time()
while True:
    img = scene.interactive_render(view, num_rays=1, max_bounces=5)
    img = cv2.resize(img, (960, 480), interpolation=cv2.INTER_NEAREST)

    fps = round(1 / (time.time() - last_time), 2)
    last_time = time.time()

    fps_label = f"fps: {fps}"
    info_label = f"pos: {np.round(view.origin, 2)}, dir: {np.round(view.dir, 2)}, fov: {math.degrees(view.fov)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, info_label, (5, 15), font, 0.5, (255, 255, 255), 1)
    cv2.putText(img, fps_label, (5, 30), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("w"):
        view.forward()
    elif key == ord("s"):
        view.backward()
    elif key == ord("a"):
        view.left()
    elif key == ord("d"):
        view.right()
    elif key == ord("i"):
        view.look_up()
    elif key == ord("k"):
        view.look_down()
    elif key == ord("j"):
        view.look_left()
    elif key == ord("l"):
        view.look_right()
