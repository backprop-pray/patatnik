from __future__ import annotations

from pathlib import Path

import pytest

from plant_pipeline.schemas.batch2 import Batch2FolderRequest, Batch2Request
from plant_pipeline.services.batch2_service import Batch2Service


def test_single_roi_returns_structured_result(batch2_config, good_roi):
    service = Batch2Service(batch2_config)
    try:
        result = service.run_batch2(Batch2Request(image_id="good-1", roi_path=str(good_roi)))
        assert result.image_id == "good-1"
        assert result.label in {"normal", "suspicious", "uncertain"}
    finally:
        service.close()


def test_missing_roi_path_fails_clearly(batch2_config):
    service = Batch2Service(batch2_config)
    try:
        with pytest.raises(FileNotFoundError):
            service.run_batch2(Batch2Request(image_id="missing", roi_path="/missing/roi.png"))
    finally:
        service.close()


def test_folder_inference_counts_failures(batch2_config, good_roi, bad_roi, tmp_path):
    folder = tmp_path / "rois"
    folder.mkdir()
    Path(folder / good_roi.name).write_bytes(good_roi.read_bytes())
    Path(folder / bad_roi.name).write_bytes(bad_roi.read_bytes())
    (folder / "broken.png").write_text("not-an-image")
    service = Batch2Service(batch2_config)
    try:
        result = service.run_batch2_folder(Batch2FolderRequest(input_dir=str(folder)))
        assert result.processed_count == 2
        assert result.failed_count == 1
    finally:
        service.close()
