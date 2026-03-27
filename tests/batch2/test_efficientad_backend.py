from __future__ import annotations

from pathlib import Path

import pytest

from plant_pipeline.anomaly.backends.efficientad_backend import EfficientAdBackend
from plant_pipeline.schemas.batch2 import Batch2Request


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
    monkeypatch.setattr(
        "plant_pipeline.anomaly.backends.efficientad_backend.predict_efficientad_paths",
        lambda *args, **kwargs: [{"image_path": str(good_roi), "score": 0.82, "anomaly_map": None, "checkpoint_hparams": {}}],
    )
    backend = EfficientAdBackend(efficientad_config)
    result = backend.predict(Batch2Request(image_id="img-1", roi_path=str(good_roi)))
    assert result.image_id == "img-1"
    assert result.model_name == efficientad_config.efficientad.model_name
    assert result.model_version == efficientad_config.efficientad.model_version
    assert result.debug["backend_mode"] == "anomalib_efficientad"
    backend.close()
