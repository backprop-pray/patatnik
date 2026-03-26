from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import DataLoader

from plant_pipeline.anomaly.base import AnomalyBackend
from plant_pipeline.anomaly.bundle import load_model_bundle
from plant_pipeline.config.settings import Batch2Config
from plant_pipeline.schemas.batch2 import Batch2FolderRequest, Batch2FolderResult, Batch2Request, SuspicionResult


def _resolve_lightning_accelerator(device: str) -> str:
    normalized = device.lower()
    if normalized in {"cpu", "cuda", "gpu", "mps", "auto"}:
        return "cuda" if normalized == "gpu" else normalized
    return "cpu"


def _load_efficientad_runtime() -> dict[str, Any]:
    data_module = importlib.import_module("anomalib.data")
    data_utils_module = importlib.import_module("anomalib.data.utils")
    efficient_module = importlib.import_module("anomalib.models.efficient_ad.lightning_model")
    torch_model_module = importlib.import_module("anomalib.models.efficient_ad.torch_model")
    pl_module = importlib.import_module("pytorch_lightning")
    torch_module = importlib.import_module("torch")
    albumentations = importlib.import_module("albumentations")
    to_tensor = importlib.import_module("albumentations.pytorch")
    torchvision_datasets = importlib.import_module("torchvision.datasets")
    return {
        "torch": torch_module,
        "pl": pl_module,
        "A": albumentations,
        "ToTensorV2": getattr(to_tensor, "ToTensorV2"),
        "ImageFolder": getattr(torchvision_datasets, "ImageFolder"),
        "download_and_extract": getattr(data_utils_module, "download_and_extract"),
        "DownloadInfo": getattr(data_utils_module, "DownloadInfo"),
        "InferenceDataset": getattr(data_module, "InferenceDataset"),
        "InputNormalizationMethod": getattr(data_utils_module, "InputNormalizationMethod"),
        "get_transforms": getattr(data_utils_module, "get_transforms"),
        "EfficientAd": getattr(efficient_module, "EfficientAd"),
        "TransformsWrapper": getattr(efficient_module, "TransformsWrapper"),
        "IMAGENETTE_DOWNLOAD_INFO": getattr(efficient_module, "IMAGENETTE_DOWNLOAD_INFO"),
        "WEIGHTS_DOWNLOAD_INFO": getattr(efficient_module, "WEIGHTS_DOWNLOAD_INFO"),
        "EfficientAdModelSize": getattr(torch_model_module, "EfficientAdModelSize"),
    }


def _build_repo_efficientad_class(runtime: dict[str, Any]):
    EfficientAd = runtime["EfficientAd"]
    download_and_extract = runtime["download_and_extract"]
    weights_info = runtime["WEIGHTS_DOWNLOAD_INFO"]
    imagenette_info = runtime["IMAGENETTE_DOWNLOAD_INFO"]
    torch_module = runtime["torch"]
    A = runtime["A"]
    ToTensorV2 = runtime["ToTensorV2"]
    ImageFolder = runtime["ImageFolder"]
    TransformsWrapper = runtime["TransformsWrapper"]

    class RepoEfficientAd(EfficientAd):
        def __init__(self, *args, teacher_weights_dir: str, imagenette_dir: str, **kwargs) -> None:
            self._teacher_weights_dir = Path(teacher_weights_dir)
            self._imagenette_dir = Path(imagenette_dir)
            super().__init__(*args, **kwargs)

        def prepare_pretrained_model(self) -> None:
            pretrained_models_dir = self._teacher_weights_dir
            if not pretrained_models_dir.is_dir():
                download_and_extract(pretrained_models_dir, weights_info)
            teacher_path = pretrained_models_dir / "efficientad_pretrained_weights" / f"pretrained_teacher_{self.model_size}.pth"
            self.model.teacher.load_state_dict(torch_module.load(teacher_path, map_location=torch_module.device(self.device)))

        def prepare_imagenette_data(self) -> None:
            self.data_transforms_imagenet = A.Compose(
                [
                    A.Resize(self.image_size[0] * 2, self.image_size[1] * 2),
                    A.ToGray(p=0.3),
                    A.CenterCrop(self.image_size[0], self.image_size[1]),
                    A.ToFloat(always_apply=False, p=1.0, max_value=255),
                    ToTensorV2(),
                ]
            )
            imagenet_dir = self._imagenette_dir
            if not imagenet_dir.is_dir():
                download_and_extract(imagenet_dir, imagenette_info)
            imagenet_dataset = ImageFolder(imagenet_dir, transform=TransformsWrapper(t=self.data_transforms_imagenet))
            self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            self.imagenet_iterator = iter(self.imagenet_loader)

    return RepoEfficientAd


def _coerce_model_size(raw_value: Any, runtime: dict[str, Any]):
    enum_cls = runtime["EfficientAdModelSize"]
    if isinstance(raw_value, enum_cls):
        return raw_value
    return enum_cls(raw_value)


def load_efficientad_checkpoint(checkpoint_path: str | Path, config: Batch2Config) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    runtime = _load_efficientad_runtime()
    torch_module = runtime["torch"]
    checkpoint = torch_module.load(str(checkpoint_path), map_location=config.efficientad.device, weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    RepoEfficientAd = _build_repo_efficientad_class(runtime)
    image_size = hparams.get("image_size", [config.efficientad.image_size, config.efficientad.image_size])
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)
    model = RepoEfficientAd(
        teacher_out_channels=hparams.get("teacher_out_channels", config.efficientad.teacher_out_channels),
        image_size=image_size,
        model_size=_coerce_model_size(hparams.get("model_size", config.efficientad.model_size), runtime),
        lr=hparams.get("lr", config.efficientad.lr),
        weight_decay=hparams.get("weight_decay", config.efficientad.weight_decay),
        padding=hparams.get("padding", config.efficientad.padding),
        pad_maps=hparams.get("pad_maps", config.efficientad.pad_maps),
        batch_size=hparams.get("batch_size", config.efficientad.train_batch_size),
        teacher_weights_dir=config.efficientad.teacher_weights_dir,
        imagenette_dir=config.efficientad.imagenette_dir,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model, checkpoint, runtime


def predict_efficientad_paths(
    checkpoint_path: str | Path,
    input_path: str | Path,
    *,
    config: Batch2Config,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> list[dict[str, Any]]:
    model, checkpoint, runtime = load_efficientad_checkpoint(checkpoint_path, config=config)
    normalization = runtime["InputNormalizationMethod"].NONE
    transform = runtime["get_transforms"](
        image_size=config.efficientad.image_size,
        center_crop=config.efficientad.center_crop,
        normalization=normalization,
    )
    normalized_input = Path(input_path)
    dataset = runtime["InferenceDataset"](path=normalized_input, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size or config.efficientad.eval_batch_size,
        shuffle=False,
        num_workers=config.efficientad.num_workers if num_workers is None else num_workers,
    )
    trainer = runtime["pl"].Trainer(
        accelerator=_resolve_lightning_accelerator(config.efficientad.device),
        devices=1,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    predictions = trainer.predict(model=model, dataloaders=dataloader)
    items: list[dict[str, Any]] = []
    for batch in predictions:
        image_paths = batch["image_path"]
        pred_scores = batch["pred_scores"]
        anomaly_maps = batch.get("anomaly_maps")
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if hasattr(pred_scores, "detach"):
            pred_scores = pred_scores.detach().cpu()
        if anomaly_maps is not None and hasattr(anomaly_maps, "detach"):
            anomaly_maps = anomaly_maps.detach().cpu()
        for index, image_path in enumerate(image_paths):
            score_value = pred_scores[index]
            score = float(score_value.item()) if hasattr(score_value, "item") else float(score_value)
            anomaly_map = None
            if anomaly_maps is not None:
                anomaly_map = np.asarray(anomaly_maps[index]).squeeze().astype(np.float32)
            items.append(
                {
                    "image_path": str(image_path),
                    "score": score,
                    "anomaly_map": anomaly_map,
                    "checkpoint_hparams": checkpoint.get("hyper_parameters", {}),
                }
            )
    return items


class EfficientAdBackend(AnomalyBackend):
    name = "efficientad"

    def __init__(self, config: Batch2Config) -> None:
        self.config = config
        self.bundle = None
        self.model_name = config.efficientad.model_name
        self.model_version = config.efficientad.model_version
        self._loaded = False

    def load(self) -> None:
        self.bundle = load_model_bundle(self.config)
        checkpoint = Path(self.bundle.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"EfficientAD checkpoint not found: {checkpoint}")
        _load_efficientad_runtime()
        self.model_name = self.bundle.model_name
        self.model_version = self.bundle.model_version
        self._loaded = True

    def predict(self, request: Batch2Request) -> SuspicionResult:
        if not self._loaded:
            self.load()
        roi_path = Path(request.roi_path)
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI not found: {request.roi_path}")
        image = cv2.imread(str(roi_path))
        if image is None:
            raise ValueError(f"Failed to read ROI image: {request.roi_path}")
        items = predict_efficientad_paths(
            self.bundle.checkpoint_path,
            request.roi_path,
            config=self.config,
            batch_size=1,
            num_workers=self.config.efficientad.num_workers,
        )
        if not items:
            raise RuntimeError(f"EfficientAD returned no predictions for {request.roi_path}")
        score = float(items[0]["score"])
        anomaly_map = items[0]["anomaly_map"]
        lower, upper = self._thresholds()
        label = self._label_for_score(score, lower, upper)
        confidence = self._confidence_for_score(score, lower, upper)
        anomaly_map_path = None
        if anomaly_map is not None and self.config.batch2.write_anomaly_map:
            anomaly_map_path = self._write_anomaly_map(request.image_id, anomaly_map)
        return SuspicionResult(
            image_id=request.image_id,
            roi_path=request.roi_path,
            label=label,
            suspicious=label == "suspicious",
            suspicious_score=score,
            confidence=confidence,
            lower_threshold=lower,
            upper_threshold=upper,
            anomaly_map_path=anomaly_map_path,
            model_name=self.model_name,
            model_version=self.model_version,
            debug={
                "backend_mode": "anomalib_efficientad",
                "bundle_dir": self.bundle.bundle_dir if self.bundle is not None else "",
                "metadata": request.metadata,
            },
        )

    def predict_folder(self, request: Batch2FolderRequest) -> Batch2FolderResult:
        if not self._loaded:
            self.load()
        input_dir = Path(request.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"ROI folder not found: {request.input_dir}")
        results = []
        failed_count = 0
        for roi_path in sorted(input_dir.glob(request.glob_pattern)):
            if not roi_path.is_file():
                continue
            image_id = roi_path.stem
            try:
                results.append(self.predict(Batch2Request(image_id=image_id, roi_path=str(roi_path), metadata=request.metadata)))
            except Exception:
                failed_count += 1
        return Batch2FolderResult(
            results=results,
            processed_count=len(results),
            failed_count=failed_count,
            debug={"input_dir": str(input_dir), "glob_pattern": request.glob_pattern},
        )

    def close(self) -> None:
        self._loaded = False

    def _thresholds(self) -> tuple[float, float]:
        if self.bundle is None:
            raise RuntimeError("EfficientAD bundle is not loaded.")
        lower = self.config.thresholds.lower_threshold
        upper = self.config.thresholds.upper_threshold
        return (
            float(self.bundle.thresholds.lower_threshold if lower is None else lower),
            float(self.bundle.thresholds.upper_threshold if upper is None else upper),
        )

    def _label_for_score(self, score: float, lower: float, upper: float) -> str:
        if score < lower:
            return "normal"
        if score > upper:
            return "suspicious"
        return "uncertain"

    def _confidence_for_score(self, score: float, lower: float, upper: float) -> float:
        midpoint = (lower + upper) / 2.0
        if score < lower:
            return float(np.clip((midpoint - score) / max(abs(midpoint), 1e-6), 0.0, 1.0))
        if score > upper:
            return float(np.clip((score - midpoint) / max(abs(score), 1e-6), 0.0, 1.0))
        half_band = max((upper - lower) / 2.0, 1e-6)
        return float(np.clip(abs(score - midpoint) / half_band, 0.0, 1.0) * 0.5)

    def _write_anomaly_map(self, image_id: str, anomaly_map: np.ndarray) -> str:
        output_dir = Path(self.config.batch2.output_root) / image_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"anomaly_map.{self.config.batch2.anomaly_map_format}"
        normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_path), colored)
        return str(output_path)
