#!/home/yasen/.venv/bin/python3
"""
Rover camera server.
Captures frames from PiCam2, reads ultrasonic sensors,
and streams both over a TCP socket to the viewer on the Mac.

Protocol per frame:
  [4 bytes] metadata JSON length  (big-endian uint32)
  [N bytes] JSON  {"L":float, "F":float, "R":float}
  [4 bytes] JPEG length           (big-endian uint32)
  [M bytes] JPEG bytes
"""

import argparse
import json
import os
import socket
import struct
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import RoverAPI
from vision import VisionPipeline

HOST = '0.0.0.0'
PORT = 5000
JPEG_QUALITY = 70
FRAME_W, FRAME_H = 640, 480

DATASET_DIR = '/home/yasen/patatnik/dataset'
COLLECT_COOLDOWN = 3.0   # seconds between saved samples
_PLANT_KWS = (
    'plant', 'flower', 'shrub', 'potted', 'vase',
    'banana', 'apple', 'orange', 'broccoli', 'carrot',
    'fruit', 'vegetable', 'crop', 'vine', 'leaf', 'grass',
    'bush', 'weed', 'herb', 'tomato', 'lettuce', 'cabbage',
    'celery', 'cucumber', 'pepper', 'pumpkin', 'squash',
    'berry', 'cherry', 'grape', 'lemon', 'mango', 'melon',
    'pear', 'pineapple', 'strawberry', 'watermelon',
)


def _is_plant(name):
    n = name.lower()
    return any(kw in n for kw in _PLANT_KWS)


def _save_sample(frame_bgr, detections, w, h):
    plant_dets = [(n, x1, y1, x2, y2) for n, _c, x1, y1, x2, y2 in detections if _is_plant(n)]
    if not plant_dets:
        return
    img_dir = os.path.join(DATASET_DIR, 'images')
    lbl_dir = os.path.join(DATASET_DIR, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    cv2.imwrite(os.path.join(img_dir, f'frame_{ts}.jpg'), frame_bgr)
    lines = []
    for name, x1, y1, x2, y2 in plant_dets:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    with open(os.path.join(lbl_dir, f'frame_{ts}.txt'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'[collect] saved frame_{ts} ({len(plant_dets)} plant dets)')



# Colour per class type for bounding box overlay
_BOX_COLOURS = {
    'plant': (0, 255, 0),
    'tree': (0, 200, 80),
    'hazard': (0, 0, 255),
}

def _box_colour(name):
    n = name.lower()
    if any(kw in n for kw in (
        'plant', 'flower', 'shrub', 'potted', 'vase',
        'banana', 'apple', 'orange', 'broccoli', 'carrot',
        'fruit', 'vegetable', 'crop', 'vine', 'leaf', 'grass',
        'bush', 'weed', 'herb', 'tomato', 'lettuce', 'cabbage',
        'celery', 'cucumber', 'pepper', 'pumpkin', 'squash',
        'berry', 'cherry', 'grape', 'lemon', 'mango', 'melon',
        'pear', 'pineapple', 'strawberry', 'watermelon',
    )):
        return _BOX_COLOURS['plant']
    if 'tree' in n:
        return _BOX_COLOURS['tree']
    return _BOX_COLOURS['hazard']


def _draw_detections(frame_bgr, detections):
    for name, conf, x1, y1, x2, y2 in detections:
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        colour = _box_colour(name)
        cv2.rectangle(frame_bgr, (x1i, y1i), (x2i, y2i), colour, 2)
        label = f'{name} {conf:.0%}'
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        ty = max(y1i - 4, lh + 4)
        cv2.rectangle(frame_bgr, (x1i, ty - lh - 4), (x1i + lw + 4, ty + 2), colour, -1)
        cv2.putText(frame_bgr, label, (x1i + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true', help='Save frames+labels for training when plants are detected')
    args = parser.parse_args()

    rover = RoverAPI()
    vision = VisionPipeline('/home/yasen/patatnik/embedded/yolov8n.pt')

    if args.collect:
        print(f'[collect] Dataset collection ON  →  {DATASET_DIR}')

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f'Camera server listening on {HOST}:{PORT} …')

    try:
        while True:
            conn, addr = server.accept()
            print(f'Viewer connected from {addr}')
            try:
                _stream(rover, conn, vision, collect=args.collect)
            except (BrokenPipeError, ConnectionResetError):
                print('Viewer disconnected.')
            finally:
                conn.close()
    finally:
        rover.close()
        server.close()


def _stream(rover, conn, vision, collect=False):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    last_collect_ts = 0.0

    while True:
        t0 = time.monotonic()

        # Sensors
        raw = rover.get_ultrasonic()
        # None = timeout (nothing in range); display as max range, not 0
        _s = lambda v: v if v is not None else 300.0
        L = _s(raw.get(1))
        F = _s(raw.get(3))
        R = _s(raw.get(2))

        # Frame — flip 180° at source so YOLO sees correctly-oriented image
        frame_rgb = rover.getframe()
        frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (FRAME_W, FRAME_H))
        frame_bgr = cv2.flip(frame_bgr, -1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # YOLO detections overlay
        vision.update(frame_rgb)
        dets, dw, dh = vision.get_detections()

        # Collect training sample (respecting cooldown)
        if collect and dets:
            now = time.monotonic()
            if now - last_collect_ts >= COLLECT_COOLDOWN:
                _save_sample(frame_bgr.copy(), dets, FRAME_W, FRAME_H)
                last_collect_ts = now

        if dets:
            _draw_detections(frame_bgr, dets)

        # Encode
        _, jpeg = cv2.imencode('.jpg', frame_bgr, encode_params)
        jpeg_bytes = jpeg.tobytes()

        meta = json.dumps({'L': round(L, 1), 'F': round(F, 1), 'R': round(R, 1)}).encode()

        # Send
        conn.sendall(struct.pack('>I', len(meta)))
        conn.sendall(meta)
        conn.sendall(struct.pack('>I', len(jpeg_bytes)))
        conn.sendall(jpeg_bytes)

        elapsed = time.monotonic() - t0
        sleep_for = max(0.0, (1 / 15) - elapsed)   # ~15 fps
        time.sleep(sleep_for)


if __name__ == '__main__':
    raise SystemExit(main())
