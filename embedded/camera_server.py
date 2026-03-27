#!/usr/bin/env python3
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

HOST = '0.0.0.0'
PORT = 5000
JPEG_QUALITY = 70
FRAME_W, FRAME_H = 640, 480


def main():
    rover = RoverAPI()

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
                _stream(rover, conn)
            except (BrokenPipeError, ConnectionResetError):
                print('Viewer disconnected.')
            finally:
                conn.close()
    finally:
        rover.close()
        server.close()


def _stream(rover, conn):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        t0 = time.monotonic()

        # Sensors
        raw = rover.get_ultrasonic()
        L = raw.get(1) or 0.0
        F = raw.get(3) or 0.0
        R = raw.get(2) or 0.0

        # Frame
        frame_rgb = rover.getframe()
        frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (FRAME_W, FRAME_H))

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
