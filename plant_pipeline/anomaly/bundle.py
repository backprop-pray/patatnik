from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plant_pipeline.config.settings import Batch2Config
from plant_pipeline.schemas.batch2 import Batch2ModelBundle, ThresholdBundle


def active_backend_name(config: Batch2Config) -> str:
    return config.batch2.backend.lower()


def active_backend_settings(config: Batch2Config) -> Any:
    backend = active_backend_name(config)
    if backend == "efficientad":
        return config.efficientad
    if backend == "patchcore":
        return config.patchcore
    raise ValueError(f"Unsupported Batch 2 backend: {config.batch2.backend}")


def resolve_bundle_dir(config: Batch2Config) -> Path:
    settings = active_backend_settings(config)
    return Path(settings.bundle_root) / settings.model_version


def load_threshold_bundle(path: Path) -> ThresholdBundle:
    if not path.exists():
        raise FileNotFoundError(f"Threshold metadata not found: {path}")
    return ThresholdBundle.model_validate(json.loads(path.read_text()))


def load_model_bundle(config: Batch2Config) -> Batch2ModelBundle:
    bundle_dir = resolve_bundle_dir(config)
    settings = active_backend_settings(config)
    metadata_path = Path(settings.metadata_path) if settings.metadata_path else bundle_dir / "bundle.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Bundle metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text())
    artifact_format = payload.get("artifact_format", "anomalib_ckpt")

    def _resolve_optional_path(raw_value: str | None, default_path: Path | None = None) -> Path | None:
        candidate = raw_value if raw_value not in {"", None} else default_path
        if candidate is None:
            return None
        path = Path(candidate)
        if not path.is_absolute() and not path.exists():
            path = metadata_path.parent / path
        return path

    checkpoint_path = _resolve_optional_path(
        settings.checkpoint_path if settings.checkpoint_path else payload.get("checkpoint_path"),
        bundle_dir / "model.ckpt" if artifact_format == "anomalib_ckpt" else None,
    )
    thresholds_path = _resolve_optional_path(payload.get("thresholds_path"), bundle_dir / "thresholds.json")
    if thresholds_path is None:
        raise FileNotFoundError(f"Threshold metadata not found: {bundle_dir / 'thresholds.json'}")
    thresholds = load_threshold_bundle(thresholds_path)
    bundle = Batch2ModelBundle(
        bundle_dir=str(bundle_dir),
        model_name=payload.get("model_name", settings.model_name),
        model_version=payload.get("model_version", settings.model_version),
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        metadata_path=str(metadata_path),
        thresholds=thresholds,
        artifact_format=artifact_format,
        teacher_path=str(_resolve_optional_path(payload.get("teacher_path"))) if payload.get("teacher_path") else None,
        student_path=str(_resolve_optional_path(payload.get("student_path"))) if payload.get("student_path") else None,
        autoencoder_path=str(_resolve_optional_path(payload.get("autoencoder_path"))) if payload.get("autoencoder_path") else None,
        normalization_stats_path=(
            str(_resolve_optional_path(payload.get("normalization_stats_path")))
            if payload.get("normalization_stats_path")
            else None
        ),
    )
    return bundle


def write_model_bundle_metadata(
    bundle_dir: Path,
    *,
    model_name: str,
    model_version: str,
    image_size: int,
    dataset_version: str,
    anomalib_version: str,
    checkpoint_path: Path | None,
    thresholds_path: Path,
    calibration_mode: str | None = None,
    score_summary: dict[str, float] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = bundle_dir / "bundle.json"
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "image_size": image_size,
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "dataset_version": dataset_version,
        "anomalib_version": anomalib_version,
        "thresholds_path": str(thresholds_path),
    }
    if checkpoint_path is not None:
        payload["checkpoint_path"] = str(checkpoint_path)
    if calibration_mode is not None:
        payload["calibration_mode"] = calibration_mode
    if score_summary is not None:
        payload["score_summary"] = score_summary
    if extra_metadata:
        payload.update(extra_metadata)
    metadata_path.write_text(json.dumps(payload, indent=2))
    return metadata_path
