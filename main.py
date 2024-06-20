import time

import cv2
import numpy as np

from scene import Scene
from triangle import Triangle
from view import View

triangle = Triangle(
    points=np.array([[0.0, 0.0, 1], [1.0, 0, 0], [0, 5, 0]]),
    color=np.array([0, 0, 255]),
    material="diffuse",
)

view = View(
    origin=np.array([0, -1.0, 2.0]),
    dir=np.array([0, 0.2, .02]),
    width=160,
    height=80,
    fov=90,
)

scene = Scene([triangle])



last_time = time.time()
while True:
    img = scene.render(view, max_bounces=3)
    img = cv2.resize(img, (640, 320), interpolation=cv2.INTER_NEAREST)
    fps = round(1 / (time.time() - last_time), 2)
    last_time = time.time()
    cv2.putText(img, str(fps), (5, 12), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("w"):
        print("forward")
        view.origin[1] += 0.1
    elif key == ord("s"):
        print("backward")
        view.origin[1] -= 0.1
    elif key == ord("a"):
        print("left")
        view.origin[0] -= 0.1
    elif key == ord("d"):
        print("right")
        view.origin[0] += 0.1
    view.reset_rays()
