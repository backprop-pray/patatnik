from __future__ import annotations

import json
from pathlib import Path

import pytest

from plant_pipeline.anomaly.dataset import (
    ensure_dataset_layout,
    ingest_rois,
    install_general_plant_dataset,
    load_dataset_manifest,
    validate_dataset_layout,
)


def test_dataset_helper_creates_required_layout(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    ensure_dataset_layout(dataset_root)
    validate_dataset_layout(dataset_root)


def test_ingest_rois_uses_unique_stable_names_and_manifest(tmp_path: Path):
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "leaf.png").write_bytes(b"a")
    (source_b / "leaf.png").write_bytes(b"b")
    dataset_root = tmp_path / "dataset"
    ensure_dataset_layout(dataset_root)

    written_a = ingest_rois(source_a, dataset_root, "val", "bad", mode="copy", source_tag="class_a")
    written_b = ingest_rois(source_b, dataset_root, "val", "bad", mode="copy", source_tag="class_b")

    assert written_a[0].name != written_b[0].name
    manifest = load_dataset_manifest(dataset_root)
    assert manifest["naming_policy"] == "<source-tag>__<sha1-12>__<original-name>"
    assert len(manifest["entries"]) == 2
    assert manifest["split_counts"]["val/bad"] == 2
    assert written_a[0].suffix == ".png"
    assert written_b[0].suffix == ".png"


def test_validate_dataset_rejects_incomplete_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        validate_dataset_layout(tmp_path / "missing")


def test_install_general_plant_dataset_builds_expected_splits(efficientad_config, tmp_path: Path):
    plantvillage_root = Path(efficientad_config.efficientad.plantvillage_dir) / "raw" / "color"
    plantdoc_root = Path(efficientad_config.efficientad.plantdoc_dir)
    for directory in [
        plantvillage_root / "Apple___healthy",
        plantvillage_root / "Apple___scab",
        plantdoc_root / "train" / "Tomato leaf bacterial spot",
        plantdoc_root / "test" / "Tomato leaf bacterial spot",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    for idx in range(10):
        (plantvillage_root / "Apple___healthy" / f"healthy_{idx}.jpg").write_bytes(b"h")
        (plantvillage_root / "Apple___scab" / f"bad_{idx}.jpg").write_bytes(b"b")
        (plantdoc_root / "train" / "Tomato leaf bacterial spot" / f"train_bad_{idx}.jpg").write_bytes(b"t")
        (plantdoc_root / "test" / "Tomato leaf bacterial spot" / f"test_bad_{idx}.jpg").write_bytes(b"u")

    manifest = install_general_plant_dataset(efficientad_config.efficientad)
    assert manifest["split_counts"]["train/good"] > 0
    assert manifest["split_counts"]["val/good"] > 0
    assert manifest["split_counts"]["test/good"] > 0
    assert manifest["split_counts"]["val/bad"] > 0
    assert manifest["split_counts"]["test/bad"] > 0
    assert len(load_dataset_manifest(Path(efficientad_config.efficientad.dataset_root))["entries"]) > 0
