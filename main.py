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
    dir=np.array([0, 0.2, 0.02]),
    width=160,
    height=80,
    fov=90,
)

scene = Scene([triangle])

last_time = time.time()
while True:
    img = scene.render(view, max_bounces=3)
    img = cv2.resize(img, (960, 480), interpolation=cv2.INTER_NEAREST)
    fps = round(1 / (time.time() - last_time), 2)
    last_time = time.time()
    cv2.putText(
        img, str(fps), (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("w"):
        print("forward")
        view.forward()
    elif key == ord("s"):
        print("backward")
        view.backward()
    elif key == ord("a"):
        print("left")
        view.left()
    elif key == ord("d"):
        print("right")
        view.right()
    elif key == ord("i"):
        print("look up")
        view.look_up()
    elif key == ord("k"):
        print("look down")
        view.look_down()
    elif key == ord("j"):
        print("look left")
        view.look_left()
    elif key == ord("l"):
        print("look right")
        view.look_right()
    view.reset_rays()

