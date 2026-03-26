from __future__ import annotations

import shutil
from pathlib import Path

from plant_pipeline.config.settings import Batch2PatchCoreSettings


REQUIRED_SPLITS = {
    ("train", "good"),
    ("val", "good"),
    ("val", "bad"),
    ("test", "good"),
    ("test", "bad"),
}


def ensure_dataset_layout(dataset_root: Path) -> None:
    for split, label in REQUIRED_SPLITS:
        (dataset_root / split / label).mkdir(parents=True, exist_ok=True)


def validate_dataset_layout(dataset_root: Path) -> None:
    missing = [str(dataset_root / split / label) for split, label in REQUIRED_SPLITS if not (dataset_root / split / label).exists()]
    if missing:
        raise FileNotFoundError(f"Dataset layout is incomplete: {missing}")


def ingest_rois(source_dir: Path, dataset_root: Path, target_split: str, target_label: str, mode: str = "symlink") -> list[Path]:
    target_dir = dataset_root / target_split / target_label
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for roi_path in sorted(source_dir.glob("*")):
        if not roi_path.is_file():
            continue
        if roi_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        destination = target_dir / roi_path.name
        if destination.exists():
            destination.unlink()
        if mode == "symlink":
            destination.symlink_to(roi_path.resolve())
        elif mode == "copy":
            shutil.copy2(roi_path, destination)
        else:
            raise ValueError(f"Unknown ingest mode: {mode}")
        written.append(destination)
    return written


def dataset_paths(settings: Batch2PatchCoreSettings) -> dict[str, Path]:
    return {
        "dataset_root": Path(settings.dataset_root),
        "train_good": Path(settings.normal_train_dir),
        "val_good": Path(settings.val_good_dir),
        "val_bad": Path(settings.val_bad_dir),
        "test_good": Path(settings.test_good_dir),
        "test_bad": Path(settings.test_bad_dir),
    }
