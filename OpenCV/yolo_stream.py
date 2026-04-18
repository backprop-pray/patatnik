#!/usr/bin/env python3
"""YOLO live camera stream served as MJPEG over HTTP."""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/yasen/yolov8n.pt"
CAMERA_DEVICE = "/dev/video0"
PORT = 8080
CONF_THRESHOLD = 0.4

model = YOLO(MODEL_PATH)

latest_frame = None
frame_lock = threading.Lock()


def capture_loop():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: cannot open camera {CAMERA_DEVICE}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera opened. Streaming on http://0.0.0.0:{PORT}/")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        annotated = results.plot()

        ret2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret2:
            with frame_lock:
                latest_frame = buf.tobytes()

    cap.release()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/":
            page = b"""<html><body style="margin:0;background:#000">
<img src="/stream" style="max-width:100%;height:auto">
</body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    print("Waiting for first frame...")
    timeout = 10
    start = time.time()
    while latest_frame is None:
        if time.time() - start > timeout:
            print("Timeout waiting for frame — check camera device.")
            break
        time.sleep(0.1)

    server = HTTPServer(("0.0.0.0", PORT), StreamHandler)
    print(f"Open http://172.20.10.9:{PORT}/ in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
