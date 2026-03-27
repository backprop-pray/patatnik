from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
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


def list_image_paths(input_path: str | Path) -> list[Path]:
    root = Path(input_path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return sorted(path for path in paths if path.is_file())


def _build_default_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_raw_efficientad_bundle(bundle, *, device: str) -> dict[str, Any]:
    required_paths = {
        "teacher_path": bundle.teacher_path,
        "student_path": bundle.student_path,
        "autoencoder_path": bundle.autoencoder_path,
        "normalization_stats_path": bundle.normalization_stats_path,
    }
    missing_fields = [name for name, value in required_paths.items() if not value]
    if missing_fields:
        raise FileNotFoundError(f"EfficientAD raw bundle is missing metadata fields: {', '.join(sorted(missing_fields))}")
    missing_files = [name for name, value in required_paths.items() if not Path(value).exists()]
    if missing_files:
        raise FileNotFoundError(f"EfficientAD raw bundle files not found: {', '.join(sorted(missing_files))}")

    map_location = torch.device(device)
    teacher = torch.load(bundle.teacher_path, map_location=map_location, weights_only=False)
    student = torch.load(bundle.student_path, map_location=map_location, weights_only=False)
    autoencoder = torch.load(bundle.autoencoder_path, map_location=map_location, weights_only=False)
    stats = torch.load(bundle.normalization_stats_path, map_location=map_location, weights_only=False)

    for model in (teacher, student, autoencoder):
        model.eval()
        model.to(map_location)

    return {
        "teacher": teacher,
        "student": student,
        "autoencoder": autoencoder,
        "teacher_mean": stats["teacher_mean"].to(map_location),
        "teacher_std": stats["teacher_std"].to(map_location),
        "q_st_start": stats["q_st_start"].to(map_location),
        "q_st_end": stats["q_st_end"].to(map_location),
        "q_ae_start": stats["q_ae_start"].to(map_location),
        "q_ae_end": stats["q_ae_end"].to(map_location),
        "image_size": int(stats.get("image_size", 256)),
        "device": map_location,
    }


@torch.no_grad()
def _predict_raw_efficientad_paths(raw_bundle: dict[str, Any], input_path: str | Path) -> list[dict[str, Any]]:
    transform = _build_default_transform(raw_bundle["image_size"])
    items: list[dict[str, Any]] = []
    for image_path in list_image_paths(input_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read ROI image: {image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = rgb.shape[:2]
        tensor = transform(rgb).unsqueeze(0).to(raw_bundle["device"])

        teacher_output = raw_bundle["teacher"](tensor)
        teacher_output = (teacher_output - raw_bundle["teacher_mean"]) / raw_bundle["teacher_std"]
        student_output = raw_bundle["student"](tensor)
        autoencoder_output = raw_bundle["autoencoder"](tensor)

        map_st = torch.mean((teacher_output - student_output[:, : teacher_output.shape[1]]) ** 2, dim=1, keepdim=True)
        map_ae = torch.mean((autoencoder_output - student_output[:, teacher_output.shape[1] :]) ** 2, dim=1, keepdim=True)
        map_st = 0.1 * (map_st - raw_bundle["q_st_start"]) / (raw_bundle["q_st_end"] - raw_bundle["q_st_start"])
        map_ae = 0.1 * (map_ae - raw_bundle["q_ae_start"]) / (raw_bundle["q_ae_end"] - raw_bundle["q_ae_start"])
        map_combined = 0.5 * map_st + 0.5 * map_ae
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode="bilinear")
        anomaly_map = map_combined[0, 0].detach().cpu().numpy().astype(np.float32)
        items.append(
            {
                "image_path": str(image_path),
                "score": float(np.max(anomaly_map)),
                "anomaly_map": anomaly_map,
                "checkpoint_hparams": {},
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
        self._raw_bundle: dict[str, Any] | None = None
        self._loaded = False

    def load(self) -> None:
        self.bundle = load_model_bundle(self.config)
        if self.bundle.artifact_format == "efficientad_raw_triplet":
            self._raw_bundle = load_raw_efficientad_bundle(self.bundle, device=self.config.efficientad.device)
        else:
            checkpoint = Path(self.bundle.checkpoint_path or "")
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
        if self.bundle is None:
            raise RuntimeError("EfficientAD bundle is not loaded.")
        image = cv2.imread(str(roi_path))
        if image is None:
            raise ValueError(f"Failed to read ROI image: {request.roi_path}")
        if self.config.efficientad.use_deterministic_demo_scorer and self.config.efficientad.deterministic_enabled:
            score, anomaly_map, backend_mode, debug_metrics = self._predict_deterministic(image)
        elif self.bundle.artifact_format == "efficientad_raw_triplet":
            if self._raw_bundle is None:
                raise RuntimeError("EfficientAD raw bundle is not loaded.")
            items = _predict_raw_efficientad_paths(self._raw_bundle, request.roi_path)
            backend_mode = "repo_efficientad_raw"
            debug_metrics = {}
        else:
            items = predict_efficientad_paths(
                self.bundle.checkpoint_path,
                request.roi_path,
                config=self.config,
                batch_size=1,
                num_workers=self.config.efficientad.num_workers,
            )
            backend_mode = "anomalib_efficientad"
            debug_metrics = {}
        if not (self.config.efficientad.use_deterministic_demo_scorer and self.config.efficientad.deterministic_enabled):
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
                "backend_mode": backend_mode,
                "bundle_dir": self.bundle.bundle_dir if self.bundle is not None else "",
                **debug_metrics,
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
        self._raw_bundle = None
        self._loaded = False

    def _predict_deterministic(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray | None, str, dict[str, Any]]:
        settings = self.config.efficientad
        image = cv2.resize(image_bgr, (settings.image_size, settings.image_size))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        green_mask = cv2.inRange(
            hsv,
            np.array([settings.green_h_min, settings.green_s_min, settings.green_v_min], dtype=np.uint8),
            np.array([settings.green_h_max, 255, 255], dtype=np.uint8),
        )
        yellow_mask = cv2.inRange(
            hsv,
            np.array([settings.yellow_h_min, settings.yellow_s_min, settings.yellow_v_min], dtype=np.uint8),
            np.array([settings.yellow_h_max, 255, 255], dtype=np.uint8),
        )
        brown_mask = cv2.inRange(
            hsv,
            np.array([settings.brown_h_min, settings.brown_s_min, 0], dtype=np.uint8),
            np.array([settings.brown_h_max, 255, settings.brown_v_max], dtype=np.uint8),
        )
        rust_mask = cv2.inRange(
            hsv,
            np.array([settings.rust_h_min, settings.rust_s_min, settings.rust_v_min], dtype=np.uint8),
            np.array([settings.rust_h_max, 255, 255], dtype=np.uint8),
        )
        necrosis_mask = cv2.inRange(
            hsv,
            np.array([0, settings.necrosis_s_min, 0], dtype=np.uint8),
            np.array([179, 255, settings.necrosis_v_max], dtype=np.uint8),
        )
        sat_mask = cv2.threshold(hsv[:, :, 1], 20, 255, cv2.THRESH_BINARY)[1]
        color_seed = cv2.bitwise_or(cv2.bitwise_or(green_mask, yellow_mask), cv2.bitwise_or(brown_mask, rust_mask))
        color_seed = cv2.bitwise_or(color_seed, cv2.bitwise_and(necrosis_mask, sat_mask))
        kernel = np.ones((5, 5), np.uint8)
        leaf_mask = cv2.morphologyEx(color_seed, cv2.MORPH_CLOSE, kernel, iterations=2)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(largest))
            if contour_area > 0:
                filled = np.zeros_like(leaf_mask)
                cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
                leaf_mask = filled

        leaf_area_ratio = float(np.count_nonzero(leaf_mask)) / float(leaf_mask.size)
        if leaf_area_ratio > 0:
            lesion_seed = cv2.bitwise_or(cv2.bitwise_or(yellow_mask, brown_mask), cv2.bitwise_or(rust_mask, necrosis_mask))
            lesion_mask = cv2.bitwise_and(lesion_seed, lesion_seed, mask=leaf_mask)
            lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        else:
            lesion_mask = np.zeros_like(leaf_mask)

        lesion_ratio = float(np.count_nonzero(lesion_mask)) / float(max(np.count_nonzero(leaf_mask), 1))
        green_ratio = float(np.count_nonzero(cv2.bitwise_and(green_mask, green_mask, mask=leaf_mask))) / float(max(np.count_nonzero(leaf_mask), 1))
        edge_density = float(np.count_nonzero(cv2.Canny(gray, 60, 140) & lesion_mask)) / float(max(np.count_nonzero(leaf_mask), 1))
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)
        warm_bias = float(np.mean(((b_channel - a_channel) > 8)[leaf_mask > 0])) if np.count_nonzero(leaf_mask) else 0.0

        lesion_score = (
            lesion_ratio * 2.6
            + edge_density * 3.5
            + max(0.0, warm_bias - 0.15) * 0.5
            + max(0.0, 0.65 - green_ratio) * 0.6
        )
        lesion_score = float(np.clip(lesion_score, 0.0, 1.0))

        weak_leaf_mask = leaf_area_ratio < settings.min_leaf_coverage_for_confident_scoring
        if leaf_area_ratio < settings.leaf_min_area_ratio:
            lesion_score = max(lesion_score, 0.30)
        elif weak_leaf_mask and settings.uncertain_on_weak_leaf_mask:
            lesion_score = float(np.clip(max(lesion_score, 0.30), 0.20, 0.44))
        else:
            if lesion_ratio <= settings.normal_max_lesion_ratio and green_ratio >= 0.50:
                lesion_score = min(lesion_score, 0.18)
            if lesion_ratio >= settings.suspicious_min_lesion_ratio:
                lesion_score = max(lesion_score, 0.55)

        lesion_heatmap = (
            lesion_mask.astype(np.float32) / 255.0 * 0.75
            + (cv2.GaussianBlur((lesion_mask > 0).astype(np.float32), (0, 0), 2.2) * 0.25)
        )
        lesion_heatmap = np.clip(lesion_heatmap, 0.0, 1.0).astype(np.float32)
        return lesion_score, lesion_heatmap, "deterministic_lesion_scorer", {
            "leaf_area_ratio": leaf_area_ratio,
            "lesion_ratio": lesion_ratio,
            "green_ratio": green_ratio,
            "edge_density": edge_density,
            "warm_bias": warm_bias,
            "weak_leaf_mask": weak_leaf_mask,
        }

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
