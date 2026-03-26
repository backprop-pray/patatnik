from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from plant_pipeline.anomaly.base import AnomalyBackend
from plant_pipeline.anomaly.bundle import load_model_bundle
from plant_pipeline.config.settings import Batch2Config
from plant_pipeline.schemas.batch2 import Batch2FolderRequest, Batch2FolderResult, Batch2Request, SuspicionResult


class PatchCoreBackend(AnomalyBackend):
    name = "patchcore"

    def __init__(self, config: Batch2Config) -> None:
        self.config = config
        self.bundle = None
        self.model_name = config.patchcore.model_name
        self.model_version = config.patchcore.model_version
        self._engine: Any = None
        self._predict_dataset_cls: Any = None
        self._loaded = False
        self._anomalib_available = False

    def load(self) -> None:
        self.bundle = load_model_bundle(self.config)
        checkpoint = Path(self.bundle.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"PatchCore checkpoint not found: {checkpoint}")
        self.model_name = self.bundle.model_name
        self.model_version = self.bundle.model_version
        try:
            engine_module = importlib.import_module("anomalib.engine")
            data_module = importlib.import_module("anomalib.data")
        except ImportError:
            self._anomalib_available = False
            self._loaded = True
            return
        self._engine = getattr(engine_module, "Engine")(accelerator=self.config.patchcore.device)
        self._predict_dataset_cls = getattr(data_module, "PredictDataset")
        self._anomalib_available = True
        self._loaded = True

    def predict(self, request: Batch2Request) -> SuspicionResult:
        if not self._loaded:
            self.load()
        roi_path = Path(request.roi_path)
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI not found: {request.roi_path}")
        image = cv2.imread(str(roi_path))
        if image is None:
            raise ValueError(f"Failed to read ROI image: {request.roi_path}")
        score, anomaly_map = self._predict_raw(request, image)
        lower, upper = self._thresholds()
        label = self._label_for_score(score, lower, upper)
        confidence = self._confidence_for_score(score, lower, upper)
        anomaly_map_path = None
        if anomaly_map is not None and self.config.batch2.write_anomaly_map:
            anomaly_map_path = self._write_anomaly_map(request.image_id, anomaly_map)
        return SuspicionResult(
            image_id=request.image_id,
            roi_path=request.roi_path,
            label=label,
            suspicious=label == "suspicious",
            suspicious_score=score,
            confidence=confidence,
            lower_threshold=lower,
            upper_threshold=upper,
            anomaly_map_path=anomaly_map_path,
            model_name=self.model_name,
            model_version=self.model_version,
            debug={
                "anomalib_available": self._anomalib_available,
                "bundle_dir": self.bundle.bundle_dir if self.bundle is not None else "",
                "metadata": request.metadata,
            },
        )

    def predict_folder(self, request: Batch2FolderRequest) -> Batch2FolderResult:
        if not self._loaded:
            self.load()
        input_dir = Path(request.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"ROI folder not found: {request.input_dir}")
        results = []
        failed_count = 0
        for roi_path in sorted(input_dir.glob(request.glob_pattern)):
            if not roi_path.is_file():
                continue
            image_id = roi_path.stem
            try:
                results.append(self.predict(Batch2Request(image_id=image_id, roi_path=str(roi_path), metadata=request.metadata)))
            except Exception:
                failed_count += 1
        return Batch2FolderResult(
            results=results,
            processed_count=len(results),
            failed_count=failed_count,
            debug={"input_dir": str(input_dir), "glob_pattern": request.glob_pattern},
        )

    def close(self) -> None:
        self._engine = None
        self._predict_dataset_cls = None
        self._loaded = False

    def _predict_raw(self, request: Batch2Request, image_bgr: np.ndarray) -> tuple[float, np.ndarray | None]:
        if self._anomalib_available and self._engine is not None and self._predict_dataset_cls is not None and self.bundle is not None:
            try:
                dataset = self._predict_dataset_cls(path=request.roi_path, image_size=self.config.patchcore.image_size)
                predictions = self._engine.predict(checkpoint_path=self.bundle.checkpoint_path, dataset=dataset)
                score = self._extract_score(predictions)
                anomaly_map = self._extract_map(predictions, image_bgr.shape[:2])
                return score, anomaly_map
            except Exception:
                pass
        return self._fallback_predict_raw(image_bgr)

    def _fallback_predict_raw(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray | None]:
        image = cv2.resize(image_bgr, (self.config.patchcore.image_size, self.config.patchcore.image_size))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        val = hsv[:, :, 2].astype(np.float32) / 255.0
        exg = (2.0 * image[:, :, 1].astype(np.float32) - image[:, :, 2].astype(np.float32) - image[:, :, 0].astype(np.float32)) / 255.0
        anomaly_map = np.clip(np.abs(exg - np.median(exg)) + np.abs(val - np.median(val)) * 0.5 + sat * 0.15, 0.0, 1.0)
        score = float(np.clip(np.percentile(anomaly_map, 97), 0.0, 1.0))
        return score, anomaly_map

    def _extract_score(self, predictions: Any) -> float:
        if not predictions:
            return 0.0
        candidate = predictions[0]
        for key in ("pred_score", "anomaly_score", "pred_scores"):
            if hasattr(candidate, key):
                value = getattr(candidate, key)
                if isinstance(value, (float, int)):
                    return float(np.clip(value, 0.0, 1.0))
                if hasattr(value, "item"):
                    return float(np.clip(value.item(), 0.0, 1.0))
        if isinstance(candidate, dict):
            for key in ("pred_score", "anomaly_score", "pred_scores"):
                if key in candidate:
                    value = candidate[key]
                    if isinstance(value, (float, int)):
                        return float(np.clip(value, 0.0, 1.0))
                    if hasattr(value, "item"):
                        return float(np.clip(value.item(), 0.0, 1.0))
        return 0.0

    def _extract_map(self, predictions: Any, target_shape: tuple[int, int]) -> np.ndarray | None:
        if not predictions:
            return None
        candidate = predictions[0]
        for key in ("anomaly_map", "pred_mask"):
            value = getattr(candidate, key, None)
            if value is None and isinstance(candidate, dict):
                value = candidate.get(key)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            if hasattr(value, "numpy"):
                value = value.numpy()
            array = np.asarray(value, dtype=np.float32).squeeze()
            if array.size == 0:
                continue
            return cv2.resize(array, (target_shape[1], target_shape[0]))
        return None

    def _thresholds(self) -> tuple[float, float]:
        if self.bundle is None:
            raise RuntimeError("PatchCore bundle is not loaded.")
        lower = self.config.thresholds.lower_threshold
        upper = self.config.thresholds.upper_threshold
        return (
            float(self.bundle.thresholds.lower_threshold if lower is None else lower),
            float(self.bundle.thresholds.upper_threshold if upper is None else upper),
        )

    def _label_for_score(self, score: float, lower: float, upper: float) -> str:
        if score < lower:
            return "normal"
        if score > upper:
            return "suspicious"
        return "uncertain"

    def _confidence_for_score(self, score: float, lower: float, upper: float) -> float:
        midpoint = (lower + upper) / 2.0
        if score < lower:
            return float(np.clip((midpoint - score) / max(midpoint, 1e-6), 0.0, 1.0))
        if score > upper:
            return float(np.clip((score - midpoint) / max(1.0 - midpoint, 1e-6), 0.0, 1.0))
        half_band = max((upper - lower) / 2.0, 1e-6)
        return float(np.clip(abs(score - midpoint) / half_band, 0.0, 1.0) * 0.5)

    def _write_anomaly_map(self, image_id: str, anomaly_map: np.ndarray) -> str:
        output_dir = Path(self.config.batch2.output_root) / image_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"anomaly_map.{self.config.batch2.anomaly_map_format}"
        normalized = np.clip(anomaly_map * 255.0, 0.0, 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_path), colored)
        return str(output_path)
