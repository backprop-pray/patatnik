from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from plant_pipeline.anomaly.backends.efficientad_backend import (
    _build_repo_efficientad_class,
    _load_efficientad_runtime,
    predict_efficientad_paths,
)
from plant_pipeline.anomaly.backends.patchcore_backend import predict_patchcore_paths
from plant_pipeline.anomaly.bundle import active_backend_settings, resolve_bundle_dir, write_model_bundle_metadata
from plant_pipeline.anomaly.calibration import calibrate_thresholds, write_threshold_bundle
from plant_pipeline.anomaly.dataset import (
    ensure_dataset_layout,
    ingest_rois,
    install_general_dataset_sources,
    install_general_plant_dataset,
    validate_dataset_layout,
)
from plant_pipeline.config.settings import Batch2Config, load_batch2_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch 2 dataset/setup utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_dataset = subparsers.add_parser("init-dataset")
    init_dataset.add_argument("--config", default=None)

    install_dataset = subparsers.add_parser("install-general-dataset")
    install_dataset.add_argument("--config", default=None)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--config", default=None)
    ingest.add_argument("--source-dir", required=True)
    ingest.add_argument("--split", required=True, choices=["train", "val", "test"])
    ingest.add_argument("--label", required=True, choices=["good", "bad"])
    ingest.add_argument("--mode", default="symlink", choices=["symlink", "copy"])

    fit = subparsers.add_parser("fit")
    fit.add_argument("--config", default=None)
    fit.add_argument("--dataset-version", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--config", default=None)
    calibrate.add_argument("--dataset-version", required=True)
    calibrate.add_argument("--val-good-dir", default=None)
    calibrate.add_argument("--val-bad-dir", default=None)

    return parser


def _relative_to_dataset_root(dataset_root: str, path: str) -> str:
    return str(Path(path).resolve().relative_to(Path(dataset_root).resolve()))


def _installed_anomalib_version() -> str:
    try:
        import anomalib
    except ImportError:
        return "unavailable"
    return getattr(anomalib, "__version__", "unknown")


def _build_patchcore_anomalib_config(config: Batch2Config, project_path: Path) -> Any:
    return OmegaConf.create(
        {
            "dataset": {
                "task": "classification",
                "image_size": config.patchcore.image_size,
                "center_crop": config.patchcore.center_crop,
                "normalization": "imagenet",
                "train_batch_size": config.patchcore.train_batch_size,
                "eval_batch_size": config.patchcore.eval_batch_size,
                "num_workers": config.patchcore.num_workers,
            },
            "model": {
                "name": config.patchcore.model_name,
                "input_size": [config.patchcore.image_size, config.patchcore.image_size],
                "backbone": config.patchcore.backbone,
                "pre_trained": True,
                "layers": config.patchcore.layers,
                "coreset_sampling_ratio": config.patchcore.coreset_sampling_ratio,
                "num_neighbors": config.patchcore.num_neighbors,
                "normalization_method": config.patchcore.normalization_method,
            },
            "metrics": {
                "image": ["F1Score", "AUROC"],
                "pixel": [],
                "threshold": {"method": "adaptive", "manual_image": None, "manual_pixel": None},
            },
            "visualization": {
                "show_images": False,
                "save_images": False,
                "log_images": False,
                "image_save_path": None,
                "mode": "simple",
            },
            "project": {
                "path": str(project_path),
                "seed": 0,
            },
            "logging": {"logger": [], "log_graph": False},
            "optimization": {"export_mode": None},
            "trainer": {
                "accelerator": config.patchcore.device,
                "devices": 1,
                "max_epochs": 1,
                "enable_checkpointing": True,
                "default_root_dir": str(project_path),
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "limit_train_batches": 1.0,
                "limit_val_batches": 1.0,
                "check_val_every_n_epoch": 1,
                "val_check_interval": 1.0,
            },
        }
    )


def _fit_patchcore_bundle(config: Batch2Config) -> Path:
    validate_dataset_layout(Path(config.patchcore.dataset_root))
    from anomalib.data import Folder, TaskType
    from anomalib.models import get_model
    from anomalib.utils.callbacks import get_callbacks

    bundle_dir = resolve_bundle_dir(config)
    project_path = bundle_dir / "_training"
    project_path.mkdir(parents=True, exist_ok=True)
    anomalib_config = _build_patchcore_anomalib_config(config, project_path)
    model = get_model(anomalib_config)
    callbacks = get_callbacks(anomalib_config)
    trainer = pl.Trainer(logger=False, callbacks=callbacks, **anomalib_config.trainer)
    datamodule = Folder(
        root=config.patchcore.dataset_root,
        normal_dir=_relative_to_dataset_root(config.patchcore.dataset_root, config.patchcore.normal_train_dir),
        abnormal_dir=_relative_to_dataset_root(config.patchcore.dataset_root, config.patchcore.val_bad_dir),
        normal_test_dir=_relative_to_dataset_root(config.patchcore.dataset_root, config.patchcore.val_good_dir),
        task=TaskType.CLASSIFICATION,
        image_size=config.patchcore.image_size,
        center_crop=config.patchcore.center_crop,
        train_batch_size=config.patchcore.train_batch_size,
        eval_batch_size=config.patchcore.eval_batch_size,
        num_workers=config.patchcore.num_workers,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
    )
    trainer.fit(model=model, datamodule=datamodule)
    trained_checkpoint = project_path / "weights" / "lightning" / "model.ckpt"
    if not trained_checkpoint.exists():
        raise FileNotFoundError(f"Expected trained checkpoint at {trained_checkpoint}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = bundle_dir / "model.ckpt"
    shutil.copy2(trained_checkpoint, output_checkpoint)
    return output_checkpoint


def _fit_efficientad_bundle(config: Batch2Config) -> Path:
    validate_dataset_layout(Path(config.efficientad.dataset_root))
    from anomalib.data import Folder, TaskType

    runtime = _load_efficientad_runtime()
    RepoEfficientAd = _build_repo_efficientad_class(runtime)

    bundle_dir = resolve_bundle_dir(config)
    project_path = bundle_dir / "_training"
    checkpoint_dir = project_path / "weights" / "lightning"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="model",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator=config.efficientad.device,
        devices=1,
        max_epochs=config.efficientad.max_epochs,
        max_steps=config.efficientad.max_steps,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        default_root_dir=str(project_path),
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        logger=False,
    )
    datamodule = Folder(
        root=config.efficientad.dataset_root,
        normal_dir=_relative_to_dataset_root(config.efficientad.dataset_root, config.efficientad.normal_train_dir),
        abnormal_dir=_relative_to_dataset_root(config.efficientad.dataset_root, config.efficientad.val_bad_dir),
        normal_test_dir=_relative_to_dataset_root(config.efficientad.dataset_root, config.efficientad.val_good_dir),
        task=TaskType.CLASSIFICATION,
        image_size=config.efficientad.image_size,
        center_crop=config.efficientad.center_crop,
        normalization="none",
        train_batch_size=config.efficientad.train_batch_size,
        eval_batch_size=config.efficientad.eval_batch_size,
        num_workers=config.efficientad.num_workers,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
        seed=config.efficientad.seed,
    )
    model = RepoEfficientAd(
        teacher_out_channels=config.efficientad.teacher_out_channels,
        image_size=(config.efficientad.image_size, config.efficientad.image_size),
        model_size=runtime["EfficientAdModelSize"](config.efficientad.model_size),
        lr=config.efficientad.lr,
        weight_decay=config.efficientad.weight_decay,
        padding=config.efficientad.padding,
        pad_maps=config.efficientad.pad_maps,
        batch_size=config.efficientad.train_batch_size,
        teacher_weights_dir=config.efficientad.teacher_weights_dir,
        imagenette_dir=config.efficientad.imagenette_dir,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trained_checkpoint = checkpoint_dir / "model.ckpt"
    if not trained_checkpoint.exists():
        raise FileNotFoundError(f"Expected trained checkpoint at {trained_checkpoint}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = bundle_dir / "model.ckpt"
    shutil.copy2(trained_checkpoint, output_checkpoint)
    return output_checkpoint


def _fit_bundle(config: Batch2Config) -> Path:
    backend = config.batch2.backend.lower()
    if backend == "efficientad":
        return _fit_efficientad_bundle(config)
    if backend == "patchcore":
        return _fit_patchcore_bundle(config)
    raise ValueError(f"Unsupported Batch 2 backend: {config.batch2.backend}")


def _collect_scores(config: Batch2Config, directory: Path) -> list[float]:
    if not directory.exists():
        return []
    bundle_dir = resolve_bundle_dir(config)
    checkpoint_path = bundle_dir / "model.ckpt"
    backend = config.batch2.backend.lower()
    if backend == "efficientad":
        items = predict_efficientad_paths(
            checkpoint_path,
            directory,
            config=config,
            batch_size=config.efficientad.eval_batch_size,
            num_workers=config.efficientad.num_workers,
        )
    elif backend == "patchcore":
        items = predict_patchcore_paths(
            checkpoint_path,
            directory,
            image_size=config.patchcore.image_size,
            center_crop=config.patchcore.center_crop,
            device=config.patchcore.device,
            batch_size=config.patchcore.eval_batch_size,
            num_workers=config.patchcore.num_workers,
        )
    else:
        raise ValueError(f"Unsupported Batch 2 backend: {config.batch2.backend}")
    return [float(item["score"]) for item in items]


def _write_bundle_metadata(config: Batch2Config, *, dataset_version: str, thresholds_path: Path, score_summary: dict[str, float], calibration_mode: str) -> Path:
    bundle_dir = resolve_bundle_dir(config)
    checkpoint_path = bundle_dir / "model.ckpt"
    if config.batch2.backend.lower() == "efficientad":
        return write_model_bundle_metadata(
            bundle_dir,
            model_name=config.efficientad.model_name,
            model_version=config.efficientad.model_version,
            image_size=config.efficientad.image_size,
            dataset_version=dataset_version,
            anomalib_version=_installed_anomalib_version(),
            checkpoint_path=checkpoint_path,
            thresholds_path=thresholds_path,
            calibration_mode=calibration_mode,
            score_summary=score_summary,
            extra_metadata={
                "model_size": config.efficientad.model_size,
                "teacher_out_channels": config.efficientad.teacher_out_channels,
                "normalization_method": config.efficientad.normalization_method,
            },
        )
    return write_model_bundle_metadata(
        bundle_dir,
        model_name=config.patchcore.model_name,
        model_version=config.patchcore.model_version,
        image_size=config.patchcore.image_size,
        dataset_version=dataset_version,
        anomalib_version=_installed_anomalib_version(),
        checkpoint_path=checkpoint_path,
        thresholds_path=thresholds_path,
        calibration_mode=calibration_mode,
        score_summary=score_summary,
        extra_metadata={
            "backbone": config.patchcore.backbone,
            "layers": config.patchcore.layers,
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    config = load_batch2_settings(getattr(args, "config", None))
    settings = active_backend_settings(config)

    if args.command == "init-dataset":
        ensure_dataset_layout(Path(settings.dataset_root))
        print(json.dumps({"dataset_root": settings.dataset_root, "status": "ok"}, indent=2))
        return

    if args.command == "install-general-dataset":
        if config.batch2.backend.lower() != "efficientad":
            raise RuntimeError("install-general-dataset is only supported for the EfficientAD backend.")
        source_info = install_general_dataset_sources(config.efficientad)
        manifest = install_general_plant_dataset(config.efficientad)
        print(
            json.dumps(
                {
                    "dataset_root": config.efficientad.dataset_root,
                    "plantvillage_dir": source_info["plantvillage_dir"],
                    "plantdoc_dir": source_info["plantdoc_dir"],
                    "split_counts": manifest["split_counts"],
                },
                indent=2,
            )
        )
        return

    if args.command == "ingest":
        written = ingest_rois(
            Path(args.source_dir),
            Path(settings.dataset_root),
            args.split,
            args.label,
            mode=args.mode,
        )
        print(json.dumps({"written_count": len(written), "target_split": args.split, "target_label": args.label}, indent=2))
        return

    if args.command == "fit":
        checkpoint_path = _fit_bundle(config)
        print(
            json.dumps(
                {
                    "backend": config.batch2.backend,
                    "bundle_dir": str(resolve_bundle_dir(config)),
                    "checkpoint_path": str(checkpoint_path),
                    "train_good_dir": settings.normal_train_dir,
                },
                indent=2,
            )
        )
        return

    val_good_dir = Path(args.val_good_dir or settings.val_good_dir)
    val_bad_dir = Path(args.val_bad_dir or settings.val_bad_dir)
    good_scores = _collect_scores(config, val_good_dir)
    bad_scores = _collect_scores(config, val_bad_dir)
    thresholds = calibrate_thresholds(good_scores, bad_scores, config.thresholds, dataset_version=args.dataset_version)
    bundle_dir = resolve_bundle_dir(config)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = bundle_dir / "model.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected fitted checkpoint at {checkpoint_path}. Run the fit command first.")
    thresholds_path = write_threshold_bundle(bundle_dir / "thresholds.json", thresholds)
    metadata_path = _write_bundle_metadata(
        config,
        dataset_version=args.dataset_version,
        thresholds_path=thresholds_path,
        score_summary=thresholds.score_summary,
        calibration_mode="bad-aware" if bad_scores else "normal-only",
    )
    print(
        json.dumps(
            {
                "backend": config.batch2.backend,
                "bundle_dir": str(bundle_dir),
                "checkpoint_path": str(checkpoint_path),
                "thresholds_path": str(thresholds_path),
                "metadata_path": str(metadata_path),
                "good_score_count": len(good_scores),
                "bad_score_count": len(bad_scores),
                "lower_threshold": thresholds.lower_threshold,
                "upper_threshold": thresholds.upper_threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
