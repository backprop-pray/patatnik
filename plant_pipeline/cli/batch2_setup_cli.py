from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from plant_pipeline.anomaly.bundle import resolve_bundle_dir, write_model_bundle_metadata
from plant_pipeline.anomaly.calibration import calibrate_thresholds, write_threshold_bundle
from plant_pipeline.anomaly.dataset import ensure_dataset_layout, ingest_rois
from plant_pipeline.config.settings import load_batch2_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch 2 dataset/setup utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_dataset = subparsers.add_parser("init-dataset")
    init_dataset.add_argument("--config", default=None)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--config", default=None)
    ingest.add_argument("--source-dir", required=True)
    ingest.add_argument("--split", required=True, choices=["train", "val", "test"])
    ingest.add_argument("--label", required=True, choices=["good", "bad"])
    ingest.add_argument("--mode", default="symlink", choices=["symlink", "copy"])

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--config", default=None)
    calibrate.add_argument("--dataset-version", required=True)
    calibrate.add_argument("--val-good-dir", default=None)
    calibrate.add_argument("--val-bad-dir", default=None)

    return parser


def _score_roi(path: Path, image_size: int) -> float:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read ROI image: {path}")
    image = cv2.resize(image, (image_size, image_size))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    exg = (2.0 * image[:, :, 1].astype(np.float32) - image[:, :, 2].astype(np.float32) - image[:, :, 0].astype(np.float32)) / 255.0
    anomaly_map = np.clip(np.abs(exg - np.median(exg)) + np.abs(val - np.median(val)) * 0.5 + sat * 0.15, 0.0, 1.0)
    return float(np.clip(np.percentile(anomaly_map, 97), 0.0, 1.0))


def _collect_scores(directory: Path, image_size: int) -> list[float]:
    scores: list[float] = []
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        scores.append(_score_roi(path, image_size))
    return scores


def main() -> None:
    args = build_parser().parse_args()
    config = load_batch2_settings(args.config)

    if args.command == "init-dataset":
        ensure_dataset_layout(Path(config.patchcore.dataset_root))
        print(json.dumps({"dataset_root": config.patchcore.dataset_root, "status": "ok"}, indent=2))
        return

    if args.command == "ingest":
        written = ingest_rois(
            Path(args.source_dir),
            Path(config.patchcore.dataset_root),
            args.split,
            args.label,
            mode=args.mode,
        )
        print(json.dumps({"written_count": len(written), "target_split": args.split, "target_label": args.label}, indent=2))
        return

    val_good_dir = Path(args.val_good_dir or config.patchcore.val_good_dir)
    val_bad_dir = Path(args.val_bad_dir or config.patchcore.val_bad_dir)
    good_scores = _collect_scores(val_good_dir, config.patchcore.image_size)
    bad_scores = _collect_scores(val_bad_dir, config.patchcore.image_size) if val_bad_dir.exists() else []
    thresholds = calibrate_thresholds(good_scores, bad_scores, config.thresholds, dataset_version=args.dataset_version)
    bundle_dir = resolve_bundle_dir(config)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = bundle_dir / "model.ckpt"
    checkpoint_path.touch(exist_ok=True)
    thresholds_path = write_threshold_bundle(bundle_dir / "thresholds.json", thresholds)
    metadata_path = write_model_bundle_metadata(
        bundle_dir,
        model_name=config.patchcore.model_name,
        model_version=config.patchcore.model_version,
        backbone=config.patchcore.backbone,
        layers=config.patchcore.layers,
        image_size=config.patchcore.image_size,
        dataset_version=args.dataset_version,
        anomalib_version="unverified-local-setup",
        checkpoint_path=checkpoint_path,
        thresholds_path=thresholds_path,
    )
    print(
        json.dumps(
            {
                "bundle_dir": str(bundle_dir),
                "thresholds_path": str(thresholds_path),
                "metadata_path": str(metadata_path),
                "good_score_count": len(good_scores),
                "bad_score_count": len(bad_scores),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
