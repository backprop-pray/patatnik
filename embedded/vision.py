"""
vision.py — Lightweight hybrid vision pipeline for the patatnik rover.

Provides:
  - YOLO-based object detection (plant, tree, hazard) every YOLO_PERIOD frames
    Runs in a background daemon thread so it never blocks the control loop.
  - OpenCV HSV crop-row segmentation every frame (fast, synchronous)

All outputs normalised to [0, 1].
"""

import threading
import numpy as np
import cv2
from typing import NamedTuple

# ── Optional YOLO import ──────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── Detection config ──────────────────────────────────────────────────────────
YOLO_PERIOD   = 5     # run YOLO every N camera frames (~3 fps at 15 fps stream)
YOLO_CONF_THR = 0.30  # minimum detection confidence to consider

# Image region boundaries (fraction of frame width)
_L_END   = 1 / 3
_R_START = 2 / 3

# HSV range for green vegetation (crops / grass)
# Tune lower/upper hue bounds for your specific field conditions.
_HSV_LOWER = np.array([25,  40,  40], dtype=np.uint8)   # yellow-green
_HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)   # green-cyan


# ── Output type ───────────────────────────────────────────────────────────────
class VisionFeatures(NamedTuple):
    plant_L:     float
    plant_C:     float
    plant_R:     float
    tree_L:      float
    tree_C:      float
    tree_R:      float
    hazard_L:    float
    hazard_C:    float
    hazard_R:    float
    crop_L:      float
    crop_R:      float
    free_path_C: float


ZERO_VISION = VisionFeatures(*(0.0,) * 12)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _region(cx_norm: float) -> int:
    """Map normalised horizontal centre → 0 (L), 1 (C), 2 (R)."""
    if cx_norm < _L_END:
        return 0
    if cx_norm < _R_START:
        return 1
    return 2


def _density(mask: np.ndarray) -> float:
    """Fraction of non-zero pixels in a binary mask."""
    return float(mask.sum()) / (mask.size * 255 + 1e-6)


# ── Pipeline class ────────────────────────────────────────────────────────────
class VisionPipeline:
    """
    Thread-safe hybrid YOLO + OpenCV pipeline.

    Usage:
        vp = VisionPipeline()
        ...
        features = vp.update(frame_rgb)   # call once per control tick
    """

    def __init__(self, model_path: str = 'yolov8n.pt'):
        self._frame_idx  = 0
        self._yolo_busy  = False
        self._lock       = threading.Lock()
        self._yolo_cache = {
            'plant':  [0.0, 0.0, 0.0],
            'tree':   [0.0, 0.0, 0.0],
            'hazard': [0.0, 0.0, 0.0],
        }

        if _YOLO_AVAILABLE:
            self._model = _YOLO(model_path, verbose=False)
            # Warm-up inference so the first real call isn't slow
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False, conf=YOLO_CONF_THR)
        else:
            self._model = None

    # ── Public entry point ────────────────────────────────────────────────────
    def update(self, frame_rgb: np.ndarray) -> VisionFeatures:
        """
        frame_rgb : (H, W, 3) uint8 numpy array in RGB colour order.
        Returns   : VisionFeatures namedtuple, all values in [0, 1].
        """
        self._frame_idx += 1
        h, w = frame_rgb.shape[:2]

        # Kick off YOLO in background if it is time and the thread is free
        if (self._model is not None
                and self._frame_idx % YOLO_PERIOD == 1
                and not self._yolo_busy):
            frame_copy = frame_rgb.copy()
            t = threading.Thread(
                target=self._run_yolo,
                args=(frame_copy, w, h),
                daemon=True,
            )
            self._yolo_busy = True
            t.start()

        # Read YOLO cache (thread-safe snapshot)
        with self._lock:
            ps = list(self._yolo_cache['plant'])
            ts = list(self._yolo_cache['tree'])
            hs = list(self._yolo_cache['hazard'])

        # OpenCV runs synchronously — it is fast
        crop_L, crop_R, free_C = self._run_opencv(frame_rgb, w, h)

        return VisionFeatures(
            plant_L=ps[0],  plant_C=ps[1],  plant_R=ps[2],
            tree_L=ts[0],   tree_C=ts[1],   tree_R=ts[2],
            hazard_L=hs[0], hazard_C=hs[1], hazard_R=hs[2],
            crop_L=crop_L,  crop_R=crop_R,  free_path_C=free_C,
        )

    # ── YOLO (background thread) ──────────────────────────────────────────────
    def _run_yolo(self, frame_rgb: np.ndarray, w: int, h: int) -> None:
        """Detect objects and update self._yolo_cache. Runs in daemon thread."""
        try:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            results   = self._model.predict(
                frame_bgr, verbose=False, conf=YOLO_CONF_THR,
            )

            new_plant  = [0.0, 0.0, 0.0]
            new_tree   = [0.0, 0.0, 0.0]
            new_hazard = [0.0, 0.0, 0.0]

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    name    = r.names[int(box.cls)].lower()
                    conf    = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx_norm = ((x1 + x2) / 2.0) / w
                    bh_norm = (y2 - y1) / h
                    # Score combines detection confidence and object size
                    score   = min(1.0, conf * bh_norm)
                    reg     = _region(cx_norm)

                    # Rewardable: plant / tree
                    if any(kw in name for kw in ('plant', 'flower', 'shrub', 'potted')):
                        new_plant[reg] = max(new_plant[reg], score)
                    if 'tree' in name:
                        new_tree[reg]  = max(new_tree[reg],  score)
                        # Trees are also counted as plants (partial credit)
                        new_plant[reg] = max(new_plant[reg], score * 0.5)

                    # Hazardous: person, furniture
                    if any(kw in name for kw in ('shoe', 'chair', 'table', 'bench',
                                                  'couch', 'sofa')):
                        new_hazard[reg] = max(new_hazard[reg], score)

            with self._lock:
                self._yolo_cache['plant']  = new_plant
                self._yolo_cache['tree']   = new_tree
                self._yolo_cache['hazard'] = new_hazard
        finally:
            self._yolo_busy = False

    # ── OpenCV crop segmentation ──────────────────────────────────────────────
    def _run_opencv(
        self, frame_rgb: np.ndarray, w: int, h: int
    ) -> tuple[float, float, float]:
        """
        HSV-threshold the bottom 60 % of the frame (ground plane).

        Returns (crop_L, crop_R, free_path_C):
          crop_L     — green-pixel density in left third   → crop presence
          crop_R     — green-pixel density in right third  → crop presence
          free_path_C — non-green density in centre strip  → open path ahead
        """
        # Restrict to ground plane (ignore sky / distant objects)
        roi  = frame_rgb[int(h * 0.4):, :]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, _HSV_LOWER, _HSV_UPPER)

        rw   = mask.shape[1]
        l    = rw // 3
        r    = 2 * rw // 3

        crop_L     = min(1.0, _density(mask[:, :l]))
        crop_R     = min(1.0, _density(mask[:, r:]))
        free_path_C = max(0.0, 1.0 - _density(mask[:, l:r]))

        return crop_L, crop_R, free_path_C
