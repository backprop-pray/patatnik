from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from plant_pipeline.anomaly.backends.efficientad_backend import EfficientAdBackend
from plant_pipeline.schemas.batch2 import Batch2Request
from plant_pipeline.schemas.batch2 import ThresholdBundle


def test_efficientad_backend_loads_valid_bundle(efficientad_config):
    backend = EfficientAdBackend(efficientad_config)
    backend.load()
    assert backend.model_name == efficientad_config.efficientad.model_name
    assert backend.model_version == efficientad_config.efficientad.model_version
    backend.close()


def test_efficientad_backend_missing_checkpoint_fails_fast(efficientad_config):
    checkpoint_path = Path(efficientad_config.efficientad.bundle_root) / efficientad_config.efficientad.model_version / "model.ckpt"
    checkpoint_path.unlink()
    backend = EfficientAdBackend(efficientad_config)
    with pytest.raises(FileNotFoundError):
        backend.load()


def test_efficientad_backend_returns_structured_result(efficientad_config, good_roi, monkeypatch):
    backend = EfficientAdBackend(efficientad_config)
    result = backend.predict(Batch2Request(image_id="img-1", roi_path=str(good_roi)))
    assert result.image_id == "img-1"
    assert result.model_name == efficientad_config.efficientad.model_name
    assert result.model_version == efficientad_config.efficientad.model_version
    assert result.debug["backend_mode"] == "deterministic_lesion_scorer"
    assert result.label == "normal"
    assert result.anomaly_map_path is not None
    assert Path(result.anomaly_map_path).exists()
    backend.close()


def test_efficientad_backend_can_use_model_path_when_demo_scorer_disabled(efficientad_config, good_roi, monkeypatch):
    efficientad_config.efficientad.use_deterministic_demo_scorer = False
    monkeypatch.setattr(
        "plant_pipeline.anomaly.backends.efficientad_backend.predict_efficientad_paths",
        lambda *args, **kwargs: [{"image_path": str(good_roi), "score": 0.82, "anomaly_map": None, "checkpoint_hparams": {}}],
    )
    backend = EfficientAdBackend(efficientad_config)
    result = backend.predict(Batch2Request(image_id="img-1-model", roi_path=str(good_roi)))
    assert result.debug["backend_mode"] == "anomalib_efficientad"
    assert result.label == "suspicious"
    backend.close()


def _write_raw_bundle(config, bundle_dir: Path, *, include_stats: bool = True) -> None:
    teacher = nn.Sequential(nn.Conv2d(3, config.efficientad.teacher_out_channels, kernel_size=1))
    student = nn.Sequential(nn.Conv2d(3, config.efficientad.teacher_out_channels * 2, kernel_size=1))
    autoencoder = nn.Sequential(nn.Conv2d(3, config.efficientad.teacher_out_channels, kernel_size=1))
    teacher_path = bundle_dir / "teacher_final.pth"
    student_path = bundle_dir / "student_final.pth"
    autoencoder_path = bundle_dir / "autoencoder_final.pth"
    stats_path = bundle_dir / "normalization_stats.pt"

    torch.save(teacher, teacher_path)
    torch.save(student, student_path)
    torch.save(autoencoder, autoencoder_path)
    if include_stats:
        torch.save(
            {
                "teacher_mean": torch.zeros((1, config.efficientad.teacher_out_channels, 1, 1)),
                "teacher_std": torch.ones((1, config.efficientad.teacher_out_channels, 1, 1)),
                "q_st_start": torch.tensor(0.0),
                "q_st_end": torch.tensor(1.0),
                "q_ae_start": torch.tensor(0.0),
                "q_ae_end": torch.tensor(1.0),
                "image_size": config.efficientad.image_size,
            },
            stats_path,
        )
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
                "artifact_format": "efficientad_raw_triplet",
                "model_name": config.efficientad.model_name,
                "model_version": config.efficientad.model_version,
                "image_size": config.efficientad.image_size,
                "created_at": "2026-03-27T00:00:00Z",
                "dataset_version": "dataset-v1",
                "anomalib_version": "repo-raw-efficientad",
                "thresholds_path": "thresholds.json",
                "teacher_path": "teacher_final.pth",
                "student_path": "student_final.pth",
                "autoencoder_path": "autoencoder_final.pth",
                "normalization_stats_path": "normalization_stats.pt",
                "model_size": config.efficientad.model_size,
                "teacher_out_channels": config.efficientad.teacher_out_channels,
            },
            indent=2,
        )
    )


def test_efficientad_raw_bundle_missing_stats_fails_fast(efficientad_config):
    bundle_dir = Path(efficientad_config.efficientad.bundle_root) / efficientad_config.efficientad.model_version
    _write_raw_bundle(efficientad_config, bundle_dir, include_stats=False)
    backend = EfficientAdBackend(efficientad_config)
    with pytest.raises(FileNotFoundError, match="normalization_stats_path"):
        backend.load()


def test_efficientad_raw_bundle_returns_structured_result(efficientad_config, good_roi):
    bundle_dir = Path(efficientad_config.efficientad.bundle_root) / efficientad_config.efficientad.model_version
    _write_raw_bundle(efficientad_config, bundle_dir, include_stats=True)
    backend = EfficientAdBackend(efficientad_config)
    backend.load()
    result = backend.predict(Batch2Request(image_id="img-raw-1", roi_path=str(good_roi)))
    assert result.image_id == "img-raw-1"
    assert result.model_name == efficientad_config.efficientad.model_name
    assert result.model_version == efficientad_config.efficientad.model_version
    assert result.debug["backend_mode"] == "deterministic_lesion_scorer"
    assert isinstance(result.suspicious_score, float)
    backend.close()


def test_efficientad_deterministic_scorer_flags_background_heavy_roi_uncertain(efficientad_config, tmp_path: Path):
    image = np.full((256, 256, 3), 180, dtype=np.uint8)
    cv2.ellipse(image, (60, 60), (25, 35), 0, 0, 360, (60, 180, 60), thickness=-1)
    path = tmp_path / "weak_leaf.png"
    cv2.imwrite(str(path), image)
    backend = EfficientAdBackend(efficientad_config)
    result = backend.predict(Batch2Request(image_id="weak-leaf", roi_path=str(path)))
    assert result.label == "uncertain"
    assert result.debug["weak_leaf_mask"] is True
    backend.close()


def test_efficientad_deterministic_scorer_flags_diseased_leaf_suspicious(efficientad_config, bad_roi):
    backend = EfficientAdBackend(efficientad_config)
    result = backend.predict(Batch2Request(image_id="bad-leaf", roi_path=str(bad_roi)))
    assert result.label == "suspicious"
    assert result.debug["lesion_ratio"] > 0
    backend.close()
