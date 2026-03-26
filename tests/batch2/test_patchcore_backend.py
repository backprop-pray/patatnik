from __future__ import annotations

from pathlib import Path

import pytest

from plant_pipeline.anomaly.backends.patchcore_backend import PatchCoreBackend
from plant_pipeline.schemas.batch2 import Batch2Request


def test_backend_loads_valid_bundle(batch2_config):
    backend = PatchCoreBackend(batch2_config)
    backend.load()
    assert backend.model_name == batch2_config.patchcore.model_name
    assert backend.model_version == batch2_config.patchcore.model_version
    backend.close()


def test_missing_checkpoint_fails_fast(batch2_config):
    checkpoint_path = Path(batch2_config.patchcore.bundle_root) / batch2_config.patchcore.model_version / "model.ckpt"
    checkpoint_path.unlink()
    backend = PatchCoreBackend(batch2_config)
    with pytest.raises(FileNotFoundError):
        backend.load()


def test_missing_thresholds_fail_fast(batch2_config):
    thresholds_path = Path(batch2_config.patchcore.bundle_root) / batch2_config.patchcore.model_version / "thresholds.json"
    thresholds_path.unlink()
    backend = PatchCoreBackend(batch2_config)
    with pytest.raises(FileNotFoundError):
        backend.load()


def test_backend_returns_structured_result(batch2_config, good_roi):
    backend = PatchCoreBackend(batch2_config)
    result = backend.predict(Batch2Request(image_id="img-1", roi_path=str(good_roi)))
    assert result.image_id == "img-1"
    assert result.model_name == batch2_config.patchcore.model_name
    assert result.model_version == batch2_config.patchcore.model_version
    backend.close()
