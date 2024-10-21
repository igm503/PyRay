import time
import math

import cv2
import numpy as np

from src import parse_yaml

scene, view = parse_yaml("scenes/default.yaml")

last_time = time.time()
while True:
    img = scene.render(view, num_rays=10, max_bounces=50, device="metal")
    img = cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_NEAREST)

    fps = round(1 / (time.time() - last_time), 2)
    last_time = time.time()

    fps_label = f"fps: {fps}"
    info_label = f"pos: {np.round(view.origin, 2)}, dir: {np.round(view.dir, 2)}, fov: {math.degrees(view.fov)}"
    print(fps_label, info_label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = len(info_label) * 9
    cv2.rectangle(img, (0, 0), (width, 40), (0, 0, 0), -1)
    cv2.putText(img, info_label, (5, 15), font, 0.5, (255, 255, 255), 1)
    cv2.putText(img, fps_label, (5, 30), font, 0.5, (255, 255, 255), 1)
    cv2.imshow("image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("w"):
        view.forward()
    elif key == ord("s"):
        view.back()
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
    elif key == ord("r"):
        view.height = 2160
        view.width = 3840
        img = scene.cumulative_render(
            view, num_rays=1000, max_bounces=500, device="metal", save_dir="output"
        )
    elif key == ord("="):
        view.fov -= 0.05 * view.fov
    elif key == ord("-"):
        view.fov += 0.05 * view.fov
