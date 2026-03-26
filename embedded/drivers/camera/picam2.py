import threading
import time

from picamera2 import Picamera2


class PiCam2FrameDriver:
    def __init__(self, size=(640, 480), pixel_format='RGB888', warmup_seconds=0.2):
        self.size = size
        self.pixel_format = pixel_format
        self.warmup_seconds = warmup_seconds

        self._camera = None
        self._running = False
        self._lock = threading.Lock()

    def _ensure_camera(self):
        if self._camera is not None:
            return

        cameras = Picamera2.global_camera_info()
        if not cameras:
            raise RuntimeError('No Pi camera detected. Check cable connection and camera interface settings.')

        self._camera = Picamera2()

    def open(self):
        with self._lock:
            if self._running:
                return

            self._ensure_camera()
            config = self._camera.create_video_configuration(
                main={'size': self.size, 'format': self.pixel_format}
            )
            self._camera.configure(config)
            self._camera.start()

            if self.warmup_seconds > 0:
                time.sleep(self.warmup_seconds)

            self._running = True

    def get_frame(self):
        if not self._running:
            self.open()

        with self._lock:
            return self._camera.capture_array()

    def close(self):
        with self._lock:
            if not self._running or self._camera is None:
                return

            self._camera.stop()
            self._running = False
