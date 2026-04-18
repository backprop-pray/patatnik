import logging
import threading
import time

import numpy as np

from picamera2 import Picamera2

log = logging.getLogger(__name__)


class PiCam2FrameDriver:
    def __init__(self, size=(640, 480), pixel_format='RGB888', warmup_seconds=1.5, vflip=False):
        self.size = size
        self.pixel_format = pixel_format
        self.warmup_seconds = warmup_seconds
        self.vflip = vflip

        self._camera = None
        self._running = False
        self._lock = threading.Lock()

    def _ensure_camera(self):
        if self._camera is not None:
            return

        cameras = Picamera2.global_camera_info()
        log.info('Available cameras: %s', cameras)
        if not cameras:
            raise RuntimeError('No camera detected. Check cable/connection.')

        self._camera = Picamera2()

    def open(self):
        with self._lock:
            if self._running:
                return

            self._ensure_camera()

            # Try video config first (works for USB), fall back to preview (Pi CSI)
            try:
                config = self._camera.create_video_configuration(
                    main={'size': self.size, 'format': self.pixel_format}
                )
                self._camera.configure(config)
                log.info('Camera configured with video configuration')
            except Exception as e:
                log.warning('Video config failed (%s), trying preview config', e)
                config = self._camera.create_preview_configuration(
                    main={'size': self.size, 'format': self.pixel_format}
                )
                self._camera.configure(config)
                log.info('Camera configured with preview configuration')

            self._camera.start()

            if self.warmup_seconds > 0:
                time.sleep(self.warmup_seconds)

            self._running = True
            log.info('Camera started at %s %s', self.size, self.pixel_format)

    def get_frame(self):
        if not self._running:
            self.open()

        with self._lock:
            frame = self._camera.capture_array('main')
            if self.vflip:
                frame = np.flipud(frame)
            return frame

    def take_picture(self):
        return self.get_frame()

    def close(self):
        with self._lock:
            if not self._running or self._camera is None:
                return
            try:
                self._camera.stop()
            except Exception:
                pass
            try:
                self._camera.close()
            except Exception:
                pass
            self._running = False
