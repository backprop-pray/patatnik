from __future__ import annotations

import json
from pathlib import Path

from plant_pipeline.config.settings import Batch2Config
from plant_pipeline.schemas.batch2 import Batch2ModelBundle, ThresholdBundle


def resolve_bundle_dir(config: Batch2Config) -> Path:
    return Path(config.patchcore.bundle_root) / config.patchcore.model_version


def load_threshold_bundle(path: Path) -> ThresholdBundle:
    if not path.exists():
        raise FileNotFoundError(f"Threshold metadata not found: {path}")
    return ThresholdBundle.model_validate(json.loads(path.read_text()))


def load_model_bundle(config: Batch2Config) -> Batch2ModelBundle:
    bundle_dir = resolve_bundle_dir(config)
    metadata_path = Path(config.patchcore.metadata_path) if config.patchcore.metadata_path else bundle_dir / "bundle.json"
    checkpoint_path = Path(config.patchcore.checkpoint_path) if config.patchcore.checkpoint_path else bundle_dir / "model.ckpt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Bundle metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text())
    thresholds_path = Path(payload.get("thresholds_path", bundle_dir / "thresholds.json"))
    thresholds = load_threshold_bundle(thresholds_path)
    bundle = Batch2ModelBundle(
        bundle_dir=str(bundle_dir),
        model_name=payload.get("model_name", config.patchcore.model_name),
        model_version=payload.get("model_version", config.patchcore.model_version),
        checkpoint_path=str(checkpoint_path),
        metadata_path=str(metadata_path),
        thresholds=thresholds,
    )
    return bundle


def write_model_bundle_metadata(
    bundle_dir: Path,
    *,
    model_name: str,
    model_version: str,
    backbone: str,
    layers: list[str],
    image_size: int,
    dataset_version: str,
    anomalib_version: str,
    checkpoint_path: Path,
    thresholds_path: Path,
) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = bundle_dir / "bundle.json"
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "backbone": backbone,
        "layers": layers,
        "image_size": image_size,
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "dataset_version": dataset_version,
        "anomalib_version": anomalib_version,
        "thresholds_path": str(thresholds_path),
        "checkpoint_path": str(checkpoint_path),
    }
    metadata_path.write_text(json.dumps(payload, indent=2))
    return metadata_path
