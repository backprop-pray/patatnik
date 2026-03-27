from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from plant_pipeline.schemas.batch1 import BoundingBox, DetectionBox


class UltralyticsLeafBackend:
    name = "ultralytics_leaf"
    license_tag = "AGPL-3.0-or-commercial"

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        if not model_path:
            raise ValueError("Ultralytics backend requires model_path.")
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Ultralytics model file not found: {self.model_path}")
        try:
            module = importlib.import_module("ultralytics")
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Install it to use the ultralytics_leaf backend.") from exc
        try:
            self._model = module.YOLO(str(model_file))
        except Exception as exc:
            raise RuntimeError(f"Failed to load Ultralytics model from {self.model_path}") from exc

    def detect(self, image_bgr: np.ndarray) -> list[DetectionBox]:
        if self._model is None:
            self.load()
        try:
            results = self._model.predict(image_bgr, save=False, verbose=False, device=self.device)
        except Exception as exc:
            raise RuntimeError(f"Ultralytics inference failed for model {self.model_name}") from exc
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        cls_values = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else list(boxes.cls)
        conf_values = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else list(boxes.conf)
        xyxy_values = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else list(boxes.xyxy)
        names = getattr(result, "names", {}) or {}

        detections: list[DetectionBox] = []
        for cls_id, conf, coords in zip(cls_values, conf_values, xyxy_values):
            label = names.get(int(cls_id), str(int(cls_id)))
            detections.append(
                DetectionBox(
                    bbox=BoundingBox(
                        x_min=int(coords[0]),
                        y_min=int(coords[1]),
                        x_max=int(coords[2]),
                        y_max=int(coords[3]),
                    ),
                    confidence=float(conf),
                    label=self._normalize_label(label),
                )
            )
        return detections

    def close(self) -> None:
        self._model = None

    @staticmethod
    def _normalize_label(label: Any) -> str:
        normalized = str(label).strip().lower()
        if normalized in {"leaf", "plant"}:
            return normalized
        return "leaf"
