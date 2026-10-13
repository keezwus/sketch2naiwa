"""PC display for the LoRA live stream: shows ORIGINAL | NAIWA side by side.
Reads data_raw/live/latest_orig.jpg + latest_lora.jpg produced by tools/live_web_lora.py.
Keys: q quit."""

import os
import socket
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(HERE, "data_raw", "live")


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


LAN = _lan_ip()
LATEST = os.path.join(LIVE, "latest_lora.jpg")
ORIG = os.path.join(LIVE, "latest_orig.jpg")
WIN = "LiveNaiwa 原图 | 奶娃 (q 退出)"

GAP = 8
H_TARGET = 512

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1100, 600)

img_a = img_b = None
mA = mB = 0.0
fps_n = 0
fps_t = time.time()
fps = 0.0
last_try = 0.0


def _load(path, last_m):
    try:
        m = os.path.getmtime(path)
    except OSError:
        return None, last_m
    if m == last_m:
        return "same", last_m
    im = None
    try:
        with open(path, "rb") as f:
            data = f.read()
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        im = None
    return im, m


def _fit_h(im, h):
    r = h / im.shape[0]
    return cv2.resize(im, (int(im.shape[1] * r), h), interpolation=cv2.INTER_AREA)


while True:
    if (time.time() - last_try) > 0.04:
        last_try = time.time()
        a, mA = _load(ORIG, mA)
        b, mB = _load(LATEST, mB)
        got = False
        if a is not None and not isinstance(a, str):
            img_a = a
            got = True
        if b is not None and not isinstance(b, str):
            img_b = b
            got = True
        if got:
            fps_n += 1

    now = time.time()
    if now - fps_t >= 1.0:
        fps = fps_n / (now - fps_t)
        fps_n = 0
        fps_t = now

    if img_a is None and img_b is None:
        disp = np.full((256, 640, 3), 28, np.uint8)
        cv2.putText(disp, "waiting for phone...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.putText(disp, f"phone: https://{LAN}:7863", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 200, 140), 1)
    else:
        left = _fit_h(img_a, H_TARGET) if img_a is not None else np.full((H_TARGET, H_TARGET, 3), 40, np.uint8)
        right = _fit_h(img_b, H_TARGET) if img_b is not None else np.full((H_TARGET, H_TARGET, 3), 40, np.uint8)
        disp = np.full((H_TARGET, left.shape[1] + GAP + right.shape[1], 3), 20, np.uint8)
        disp[:, :left.shape[1]] = left
        disp[:, left.shape[1] + GAP:] = right
        cv2.putText(disp, "original", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2)
        cv2.putText(disp, "naiwa", (left.shape[1] + GAP + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2)
    cv2.putText(disp, f"{fps:.1f} fps", (disp.shape[1] - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow(WIN, disp)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
