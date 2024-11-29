import time
import math
import argparse

import cv2
import numpy as np

from src import parse_yaml

parser = argparse.ArgumentParser()

parser.add_argument("scene", type=str, help="name of scene file (must be in scenes directory)")
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "metal", "cuda"],
    help="Device to use for rendering",
)
args = parser.parse_args()

scene_path = f"scenes/{args.scene}" if args.scene else "scenes/default.yaml"

scene, view, render_config, save_config = parse_yaml(scene_path)

debug = False
last_time = time.time()
first_time = last_time
num_frames = 0
avg_fps_start_lag = 50

while True:
    img = scene.render(
        view,
        num_rays=render_config["num_samples"],
        exposure=render_config["exposure"],
        max_bounces=render_config["max_bounces"],
        device=args.device,
    )
    num_frames += 1

    img = cv2.resize(img, render_config["display_resolution"], interpolation=cv2.INTER_NEAREST)

    if debug:
        fps = round(1 / (time.time() - last_time), 2)
        last_time = time.time()

        fps_label = f"fps: {fps}"
        info_label = f"pos: {np.round(view.origin, 2)}, dir: {np.round(view.dir, 2)}, fov: {math.degrees(view.fov)}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        width = len(info_label) * 18

        height = 140 if num_frames > avg_fps_start_lag else 100
        cv2.rectangle(img, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.putText(img, info_label, (5, 40), font, 1, (255, 255, 255), 1)
        cv2.putText(img, fps_label, (5, 80), font, 1, (255, 255, 255), 1)

        if num_frames == avg_fps_start_lag:
            first_time = time.time()
        if num_frames > avg_fps_start_lag:
            avg_fps = round((num_frames - avg_fps_start_lag) / (time.time() - first_time), 2)
            avg_fps_label = f"avg fps: {avg_fps}"
            cv2.putText(img, avg_fps_label, (5, 120), font, 1, (255, 255, 255), 1)

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
        prev_height = view.height
        prev_width = view.width
        view.width = save_config["resolution"][0]
        view.height = save_config["resolution"][1]
        scene.cumulative_render(
            view,
            num_rays=save_config["num_samples"],
            max_bounces=save_config["max_bounces"],
            save_dir=save_config["save_path"],
            exposure=save_config["exposure"],
            device=args.device,
        )
        view.height = prev_height
        view.width = prev_width
    elif key == ord("="):
        view.fov -= 0.05 * view.fov
    elif key == ord("-"):
        view.fov += 0.05 * view.fov
    elif key == ord("`"):
        debug = not debug
