from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from plant_pipeline.config.settings import Batch2EfficientAdSettings, Batch2PatchCoreSettings


REQUIRED_SPLITS = {
    ("train", "good"),
    ("val", "good"),
    ("val", "bad"),
    ("test", "good"),
    ("test", "bad"),
}

MANIFEST_NAME = "dataset_manifest.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PLANTDOC_HEALTHY_SUFFIX = " leaf"
PLANTDOC_DISEASE_KEYWORDS = (
    "scab",
    "spot",
    "rust",
    "blight",
    "mildew",
    "mosaic",
    "virus",
    "bacterial",
    "yellow",
    "rot",
    "curl",
    "mold",
)


def ensure_dataset_layout(dataset_root: Path) -> None:
    for split, label in REQUIRED_SPLITS:
        (dataset_root / split / label).mkdir(parents=True, exist_ok=True)


def validate_dataset_layout(dataset_root: Path) -> None:
    missing = [str(dataset_root / split / label) for split, label in REQUIRED_SPLITS if not (dataset_root / split / label).exists()]
    if missing:
        raise FileNotFoundError(f"Dataset layout is incomplete: {missing}")


def stable_dataset_filename(roi_path: Path, *, source_tag: str) -> str:
    digest = hashlib.sha1(str(roi_path.resolve()).encode("utf-8")).hexdigest()[:12]
    normalized_name = f"{roi_path.stem}{roi_path.suffix.lower()}"
    return f"{source_tag}__{digest}__{normalized_name}"


def _manifest_path(dataset_root: Path) -> Path:
    return dataset_root / MANIFEST_NAME


def load_dataset_manifest(dataset_root: Path) -> dict[str, Any]:
    path = _manifest_path(dataset_root)
    if not path.exists():
        return {
            "naming_policy": "<source-tag>__<sha1-12>__<original-name>",
            "entries": [],
            "split_counts": {},
        }
    return json.loads(path.read_text())


def write_dataset_manifest(dataset_root: Path, manifest: dict[str, Any]) -> Path:
    path = _manifest_path(dataset_root)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def ingest_rois(
    source_dir: Path,
    dataset_root: Path,
    target_split: str,
    target_label: str,
    mode: str = "symlink",
    *,
    source_tag: str | None = None,
) -> list[Path]:
    target_dir = dataset_root / target_split / target_label
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_dataset_manifest(dataset_root)
    written: list[Path] = []
    tag = source_tag or source_dir.name.replace(" ", "_")
    manifest_entries = manifest.setdefault("entries", [])
    split_counts = manifest.setdefault("split_counts", {})

    for roi_path in sorted(source_dir.glob("*")):
        if not roi_path.is_file():
            continue
        if roi_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        destination = target_dir / stable_dataset_filename(roi_path, source_tag=tag)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if mode == "symlink":
            destination.symlink_to(roi_path.resolve())
        elif mode == "copy":
            shutil.copy2(roi_path, destination)
        else:
            raise ValueError(f"Unknown ingest mode: {mode}")
        manifest_entries.append(
            {
                "split": target_split,
                "label": target_label,
                "output_name": destination.name,
                "source_path": str(roi_path.resolve()),
                "source_tag": tag,
                "mode": mode,
            }
        )
        written.append(destination)

    split_counts[f"{target_split}/{target_label}"] = len(list(target_dir.iterdir()))
    write_dataset_manifest(dataset_root, manifest)
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


def efficientad_dataset_paths(settings: Batch2EfficientAdSettings) -> dict[str, Path]:
    return {
        "dataset_root": Path(settings.dataset_root),
        "train_good": Path(settings.normal_train_dir),
        "val_good": Path(settings.val_good_dir),
        "val_bad": Path(settings.val_bad_dir),
        "test_good": Path(settings.test_good_dir),
        "test_bad": Path(settings.test_bad_dir),
    }


def clone_repo(
    url: str,
    destination: Path,
    *,
    ref: str = "master",
    sparse_paths: list[str] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination / ".git").exists():
        return
    clone_cmd = ["git", "clone", "--depth", "1"]
    if sparse_paths:
        clone_cmd.extend(["--filter=blob:none", "--sparse"])
    clone_cmd.extend(["--branch", ref, url, str(destination)])
    subprocess.run(clone_cmd, check=True)
    if sparse_paths:
        subprocess.run(["git", "-C", str(destination), "sparse-checkout", "set", *sparse_paths], check=True)


def install_general_dataset_sources(settings: Batch2EfficientAdSettings) -> dict[str, str]:
    clone_repo(
        "https://github.com/spMohanty/PlantVillage-Dataset.git",
        Path(settings.plantvillage_dir),
        ref="master",
        sparse_paths=["raw/color"],
    )
    clone_repo(
        "https://github.com/pratikkayal/PlantDoc-Dataset.git",
        Path(settings.plantdoc_dir),
        ref="master",
    )
    return {
        "plantvillage_dir": settings.plantvillage_dir,
        "plantdoc_dir": settings.plantdoc_dir,
    }


def _iter_image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _allocate_counts(total: int, ratios: tuple[float, ...]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def _split_paths(paths: list[Path], ratios: tuple[float, ...], *, seed: int) -> list[list[Path]]:
    items = list(paths)
    random.Random(seed).shuffle(items)
    counts = _allocate_counts(len(items), ratios)
    splits: list[list[Path]] = []
    cursor = 0
    for count in counts:
        splits.append(items[cursor : cursor + count])
        cursor += count
    return splits


def _is_plantdoc_diseased(class_name: str) -> bool:
    normalized = class_name.strip().lower()
    if normalized.endswith(PLANTDOC_HEALTHY_SUFFIX):
        return False
    return any(keyword in normalized for keyword in PLANTDOC_DISEASE_KEYWORDS)


def _record_manifest_entry(
    manifest_entries: list[dict[str, Any]],
    *,
    split: str,
    label: str,
    destination: Path,
    source_path: Path,
    source_tag: str,
    source_dataset: str,
    source_class: str,
) -> None:
    manifest_entries.append(
        {
            "split": split,
            "label": label,
            "output_name": destination.name,
            "source_path": str(source_path.resolve()),
            "source_tag": source_tag,
            "source_dataset": source_dataset,
            "source_class": source_class,
            "mode": "symlink",
        }
    )


def _write_symlink(destination: Path, source_path: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source_path.resolve())


def install_general_plant_dataset(settings: Batch2EfficientAdSettings) -> dict[str, Any]:
    dataset_root = Path(settings.dataset_root)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    ensure_dataset_layout(dataset_root)

    manifest = {
        "source_datasets": ["PlantVillage", "PlantDoc"],
        "dataset_version": settings.model_version,
        "naming_policy": "<source-tag>__<sha1-12>__<original-name>",
        "entries": [],
        "split_counts": {},
    }
    entries = manifest["entries"]

    plantvillage_root = Path(settings.plantvillage_dir) / "raw" / "color"
    if not plantvillage_root.exists():
        raise FileNotFoundError(f"PlantVillage raw/color directory not found: {plantvillage_root}")

    for class_dir in sorted(path for path in plantvillage_root.iterdir() if path.is_dir()):
        files = _iter_image_files(class_dir)
        if not files:
            continue
        source_class = class_dir.name
        source_tag = f"plantvillage_{source_class.replace(' ', '_')}"
        if "___healthy" in source_class:
            train_files, val_files, test_files = _split_paths(files, (0.8, 0.1, 0.1), seed=42)
            assignments = [("train", "good", train_files), ("val", "good", val_files), ("test", "good", test_files)]
        else:
            val_files, test_files = _split_paths(files, (0.5, 0.5), seed=42)
            assignments = [("val", "bad", val_files), ("test", "bad", test_files)]
        for split, label, split_files in assignments:
            target_dir = dataset_root / split / label
            for source_path in split_files:
                destination = target_dir / stable_dataset_filename(source_path, source_tag=source_tag)
                _write_symlink(destination, source_path)
                _record_manifest_entry(
                    entries,
                    split=split,
                    label=label,
                    destination=destination,
                    source_path=source_path,
                    source_tag=source_tag,
                    source_dataset="PlantVillage",
                    source_class=source_class,
                )

    plantdoc_root = Path(settings.plantdoc_dir)
    if not plantdoc_root.exists():
        raise FileNotFoundError(f"PlantDoc directory not found: {plantdoc_root}")

    plantdoc_by_class: dict[str, list[Path]] = {}
    for split_dir in [plantdoc_root / "train", plantdoc_root / "test"]:
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if not _is_plantdoc_diseased(class_dir.name):
                continue
            plantdoc_by_class.setdefault(class_dir.name, []).extend(_iter_image_files(class_dir))

    for source_class, files in sorted(plantdoc_by_class.items()):
        if not files:
            continue
        source_tag = f"plantdoc_{source_class.replace(' ', '_')}"
        val_files, test_files = _split_paths(files, (0.5, 0.5), seed=42)
        for split, split_files in [("val", val_files), ("test", test_files)]:
            target_dir = dataset_root / split / "bad"
            for source_path in split_files:
                destination = target_dir / stable_dataset_filename(source_path, source_tag=source_tag)
                _write_symlink(destination, source_path)
                _record_manifest_entry(
                    entries,
                    split=split,
                    label="bad",
                    destination=destination,
                    source_path=source_path,
                    source_tag=source_tag,
                    source_dataset="PlantDoc",
                    source_class=source_class,
                )

    for split, label in REQUIRED_SPLITS:
        manifest["split_counts"][f"{split}/{label}"] = len(list((dataset_root / split / label).iterdir()))
    write_dataset_manifest(dataset_root, manifest)
    return manifest
