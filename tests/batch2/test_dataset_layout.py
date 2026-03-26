from __future__ import annotations

from pathlib import Path

import pytest

from plant_pipeline.anomaly.dataset import ensure_dataset_layout, ingest_rois, validate_dataset_layout


def test_dataset_helper_creates_required_layout(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    ensure_dataset_layout(dataset_root)
    validate_dataset_layout(dataset_root)


def test_ingest_rois_symlinks_files(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    roi_path = source / "roi.png"
    roi_path.write_bytes(b"roi")
    dataset_root = tmp_path / "dataset"
    ensure_dataset_layout(dataset_root)
    written = ingest_rois(source, dataset_root, "train", "good", mode="copy")
    assert written
    assert written[0].exists()


def test_validate_dataset_rejects_incomplete_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        validate_dataset_layout(tmp_path / "missing")
