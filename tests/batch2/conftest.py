from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from plant_pipeline.config.settings import load_batch2_settings
from plant_pipeline.schemas.batch2 import ThresholdBundle


@pytest.fixture
def batch2_config(tmp_path: Path):
    config = load_batch2_settings()
    config.batch2.backend = "patchcore"
    config.batch2.output_root = str(tmp_path / "batch2-output")
    config.patchcore.bundle_root = str(tmp_path / "bundles")
    config.patchcore.dataset_root = str(tmp_path / "dataset")
    config.patchcore.normal_train_dir = str(tmp_path / "dataset" / "train" / "good")
    config.patchcore.val_good_dir = str(tmp_path / "dataset" / "val" / "good")
    config.patchcore.val_bad_dir = str(tmp_path / "dataset" / "val" / "bad")
    config.patchcore.test_good_dir = str(tmp_path / "dataset" / "test" / "good")
    config.patchcore.test_bad_dir = str(tmp_path / "dataset" / "test" / "bad")
    config.patchcore.model_version = "patchcore-test-v1"
    config.patchcore.allow_inference_fallback = True
    bundle_dir = Path(config.patchcore.bundle_root) / config.patchcore.model_version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.ckpt").write_bytes(b"test-checkpoint")
    thresholds = ThresholdBundle(
        lower_threshold=0.3,
        upper_threshold=0.7,
        normal_percentile=0.95,
        suspicious_percentile=0.995,
        calibration_dataset_version="dataset-v1",
        score_summary={"good_mean": 0.2, "bad_mean": 0.8},
    )
    (bundle_dir / "thresholds.json").write_text(json.dumps(thresholds.model_dump(mode="json"), indent=2))
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
            {
                "model_name": config.patchcore.model_name,
                "model_version": config.patchcore.model_version,
                "backbone": config.patchcore.backbone,
                "layers": config.patchcore.layers,
                "image_size": config.patchcore.image_size,
                "created_at": "2026-03-26T00:00:00Z",
                "dataset_version": "dataset-v1",
                "anomalib_version": "test",
                "thresholds_path": str(bundle_dir / "thresholds.json"),
                "checkpoint_path": str(bundle_dir / "model.ckpt"),
            },
            indent=2,
        )
    )
    return config


@pytest.fixture
def efficientad_config(tmp_path: Path):
    config = load_batch2_settings()
    config.batch2.backend = "efficientad"
    config.batch2.output_root = str(tmp_path / "batch2-output")
    config.efficientad.bundle_root = str(tmp_path / "efficientad-bundles")
    config.efficientad.dataset_root = str(tmp_path / "dataset")
    config.efficientad.normal_train_dir = str(tmp_path / "dataset" / "train" / "good")
    config.efficientad.val_good_dir = str(tmp_path / "dataset" / "val" / "good")
    config.efficientad.val_bad_dir = str(tmp_path / "dataset" / "val" / "bad")
    config.efficientad.test_good_dir = str(tmp_path / "dataset" / "test" / "good")
    config.efficientad.test_bad_dir = str(tmp_path / "dataset" / "test" / "bad")
    config.efficientad.plantvillage_dir = str(tmp_path / "external" / "PlantVillage-Dataset")
    config.efficientad.plantdoc_dir = str(tmp_path / "external" / "PlantDoc-Dataset")
    config.efficientad.teacher_weights_dir = str(tmp_path / "external" / "efficientad" / "pre_trained")
    config.efficientad.imagenette_dir = str(tmp_path / "external" / "efficientad" / "imagenette")
    config.efficientad.model_version = "efficientad-test-v1"
    bundle_dir = Path(config.efficientad.bundle_root) / config.efficientad.model_version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.ckpt").write_bytes(b"efficientad-checkpoint")
    thresholds = ThresholdBundle(
        lower_threshold=0.3,
        upper_threshold=0.7,
        normal_percentile=0.95,
        suspicious_percentile=0.995,
        calibration_dataset_version="dataset-v1",
        score_summary={"good_mean": 0.2, "bad_mean": 0.8},
    )
    (bundle_dir / "thresholds.json").write_text(json.dumps(thresholds.model_dump(mode="json"), indent=2))
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
            {
                "model_name": config.efficientad.model_name,
                "model_version": config.efficientad.model_version,
                "image_size": config.efficientad.image_size,
                "created_at": "2026-03-27T00:00:00Z",
                "dataset_version": "dataset-v1",
                "anomalib_version": "test",
                "thresholds_path": str(bundle_dir / "thresholds.json"),
                "checkpoint_path": str(bundle_dir / "model.ckpt"),
                "model_size": config.efficientad.model_size,
                "teacher_out_channels": config.efficientad.teacher_out_channels,
            },
            indent=2,
        )
    )
    return config


@pytest.fixture
def good_roi(tmp_path: Path) -> Path:
    image = np.full((256, 256, 3), (45, 160, 45), dtype=np.uint8)
    cv2.ellipse(image, (128, 128), (70, 100), 0, 0, 360, (55, 190, 55), thickness=-1)
    path = tmp_path / "good.png"
    cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def bad_roi(tmp_path: Path) -> Path:
    image = np.full((256, 256, 3), (45, 160, 45), dtype=np.uint8)
    cv2.ellipse(image, (128, 128), (70, 100), 0, 0, 360, (55, 190, 55), thickness=-1)
    cv2.rectangle(image, (40, 40), (220, 220), (0, 0, 255), thickness=20)
    path = tmp_path / "bad.png"
    cv2.imwrite(str(path), image)
    return path
