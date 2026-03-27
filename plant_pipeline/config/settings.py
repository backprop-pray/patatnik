from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    name: str = "plant-pipeline"
    environment: str = "dev"


class CaptureSettings(BaseModel):
    image_format: str = "jpg"
    working_size: int = 640
    camera_index: int = 0
    simulate: bool = True


class QualitySettings(BaseModel):
    min_blur_score: float = 90.0
    min_motion_ratio: float = 0.18
    max_motion_ratio: float = 5.5
    min_brightness: float = 45.0
    max_brightness: float = 215.0
    max_dark_fraction: float = 0.55
    max_bright_fraction: float = 0.35
    min_foreground_fraction: float = 0.03


class Batch1Settings(BaseModel):
    output_root: str = "./data/batch1"
    working_size: int = 640
    roi_format: str = "png"
    debug_overlays: bool = False
    write_roi: bool = True


class Batch1QualitySettings(BaseModel):
    min_blur_score: float = 90.0
    min_motion_ratio: float = 0.18
    max_motion_ratio: float = 5.5
    min_brightness: float = 45.0
    max_brightness: float = 215.0
    max_dark_fraction: float = 0.55
    max_bright_fraction: float = 0.35
    compute_vegetation_metrics: bool = True
    reject_on_vegetation_fraction: bool = False


class Batch1DetectorSettings(BaseModel):
    backend: str = "mock"
    model_path: str = ""
    device: str = "cpu"
    min_confidence: float = 0.15
    allowed_labels: list[str] = Field(default_factory=lambda: ["leaf", "plant"])
    warmup_runs: int = 3
    benchmark_runs: int = 20
    licensing_note: str = ""


class Batch1ClusterSettings(BaseModel):
    merge_iou_threshold: float = 0.15
    merge_distance_threshold: float = 0.08
    min_cluster_members: int = 1
    min_cluster_area_ratio: float = 0.01
    bbox_expand_ratio: float = 0.12
    single_detection_expand_ratio: float = 0.45
    single_detection_context_distance_ratio: float = 0.20
    min_final_roi_width_ratio: float = 0.20
    min_final_roi_height_ratio: float = 0.25
    oversized_cluster_penalty_start: float = 0.35
    oversized_cluster_penalty_weight: float = 0.80
    border_touch_margin_ratio: float = 0.03
    border_touch_penalty_weight: float = 0.35
    dense_scene_fallback_min_vegetation_fraction: float = 0.20
    small_cluster_fallback_min_mean_confidence: float = 0.35
    small_cluster_fallback_min_score: float = 0.05
    score_weight_confidence: float = 0.5
    score_weight_area: float = 0.3
    score_weight_centrality: float = 0.2


class Batch1ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    enable_api: bool = False


class Batch2Settings(BaseModel):
    backend: str = "efficientad"
    output_root: str = "./data/batch2"
    write_anomaly_map: bool = True
    anomaly_map_format: str = "png"
    roi_glob_pattern: str = "*.png"


class Batch2PatchCoreSettings(BaseModel):
    backend: str = "patchcore"
    dataset_root: str = "./data/datasets/roi_anomaly"
    bundle_root: str = "./data/models/batch2_patchcore"
    checkpoint_path: str = ""
    metadata_path: str = ""
    normal_train_dir: str = "./data/datasets/roi_anomaly/train/good"
    val_good_dir: str = "./data/datasets/roi_anomaly/val/good"
    val_bad_dir: str = "./data/datasets/roi_anomaly/val/bad"
    test_good_dir: str = "./data/datasets/roi_anomaly/test/good"
    test_bad_dir: str = "./data/datasets/roi_anomaly/test/bad"
    image_size: int = 224
    center_crop: Optional[int] = None
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 0
    backbone: str = "wide_resnet50_2"
    layers: list[str] = Field(default_factory=lambda: ["layer2", "layer3"])
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    normalization_method: str = "min_max"
    device: str = "cpu"
    model_name: str = "patchcore"
    model_version: str = "patchcore-wideresnet50-v1"
    allow_inference_fallback: bool = False


class Batch2EfficientAdSettings(BaseModel):
    backend: str = "efficientad"
    dataset_root: str = "./data/datasets/roi_anomaly_general"
    bundle_root: str = "./data/models/batch2_efficientad"
    checkpoint_path: str = ""
    metadata_path: str = ""
    normal_train_dir: str = "./data/datasets/roi_anomaly_general/train/good"
    val_good_dir: str = "./data/datasets/roi_anomaly_general/val/good"
    val_bad_dir: str = "./data/datasets/roi_anomaly_general/val/bad"
    test_good_dir: str = "./data/datasets/roi_anomaly_general/test/good"
    test_bad_dir: str = "./data/datasets/roi_anomaly_general/test/bad"
    external_root: str = "./data/external"
    plantvillage_dir: str = "./data/external/PlantVillage-Dataset"
    plantdoc_dir: str = "./data/external/PlantDoc-Dataset"
    teacher_weights_dir: str = "./data/external/efficientad/pre_trained"
    imagenette_dir: str = "./data/external/efficientad/imagenette"
    image_size: int = 256
    center_crop: Optional[int] = None
    train_batch_size: int = 1
    eval_batch_size: int = 16
    num_workers: int = 0
    device: str = "cpu"
    model_name: str = "efficientad"
    model_version: str = "efficientad-s-general-v1"
    model_size: str = "small"
    teacher_out_channels: int = 384
    lr: float = 0.0001
    weight_decay: float = 0.00001
    padding: bool = False
    pad_maps: bool = True
    normalization_method: str = "min_max"
    max_epochs: int = 200
    max_steps: int = 70000
    seed: int = 42


class Batch2ThresholdSettings(BaseModel):
    lower_threshold: Optional[float] = None
    upper_threshold: Optional[float] = None
    normal_percentile: float = 0.95
    suspicious_percentile: float = 0.995
    min_gap: float = 0.05
    confidence_midpoint_weight: float = 1.0
    require_bad_validation: bool = True
    min_val_good_count: int = 20
    min_val_bad_count: int = 20


class Batch2ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8001
    enable_api: bool = False


class DetectSettings(BaseModel):
    min_prefilter_fraction: float = 0.04
    confidence_threshold: float = 0.55
    fallback_low_confidence: float = 0.35
    centrality_threshold: float = 0.5
    bbox_expand_ratio: float = 0.10
    input_size: int = 640
    model_path: str = ""
    labels: list[str] = Field(default_factory=lambda: ["plant"])
    model_version: str = "efficientdet-lite0-int8-v1"


class AnomalySettings(BaseModel):
    model_version: str = "patchcore-resnet18-v1"
    backbone: str = "resnet18"
    image_size: int = 224
    normal_threshold: float = 0.35
    suspicious_threshold: float = 0.60
    memory_bank_path: str = ""
    embedding_path: str = ""


class CompressionSettings(BaseModel):
    format: str = "webp"
    jpeg_fallback: bool = True
    thumbnail_max_side: int = 256
    review_max_side: int = 1280
    roi_max_side: int = 768
    webp_quality: int = 82
    jpeg_quality: int = 88


class StorageSettings(BaseModel):
    root_dir: str = "./data"
    sqlite_path: str = "./data/pipeline.db"
    debug_artifacts: bool = False


class UploadSettings(BaseModel):
    enabled: bool = False
    endpoint: str = "http://localhost:8080/api/inspections"
    timeout_seconds: int = 10
    retry_base_seconds: int = 5
    retry_max_seconds: int = 300
    max_attempts: int = 10
    wifi_check_host: str = "8.8.8.8"
    wifi_check_port: int = 53


class LoraSettings(BaseModel):
    enabled: bool = False


class LoggingSettings(BaseModel):
    level: str = "INFO"


class PipelineSettings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    detect: DetectSettings = Field(default_factory=DetectSettings)
    anomaly: AnomalySettings = Field(default_factory=AnomalySettings)
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    upload: UploadSettings = Field(default_factory=UploadSettings)
    lora: LoraSettings = Field(default_factory=LoraSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


class Batch1Config(BaseModel):
    batch1: Batch1Settings = Field(default_factory=Batch1Settings)
    quality_batch1: Batch1QualitySettings = Field(default_factory=Batch1QualitySettings)
    detector_batch1: Batch1DetectorSettings = Field(default_factory=Batch1DetectorSettings)
    cluster: Batch1ClusterSettings = Field(default_factory=Batch1ClusterSettings)
    api: Batch1ApiSettings = Field(default_factory=Batch1ApiSettings)


class Batch2Config(BaseModel):
    batch2: Batch2Settings = Field(default_factory=Batch2Settings)
    patchcore: Batch2PatchCoreSettings = Field(default_factory=Batch2PatchCoreSettings)
    efficientad: Batch2EfficientAdSettings = Field(default_factory=Batch2EfficientAdSettings)
    thresholds: Batch2ThresholdSettings = Field(default_factory=Batch2ThresholdSettings)
    api: Batch2ApiSettings = Field(default_factory=Batch2ApiSettings)


def load_settings(path: Optional[Union[str, Path]] = None) -> PipelineSettings:
    if path is None:
        path = Path(__file__).with_name("default.yaml")
    else:
        path = Path(path)
    payload: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return PipelineSettings.model_validate(payload)


def load_batch1_settings(path: Optional[Union[str, Path]] = None) -> Batch1Config:
    if path is None:
        path = Path(__file__).with_name("batch1.yaml")
    else:
        path = Path(path)
    payload: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return Batch1Config.model_validate(payload)


def load_batch2_settings(path: Optional[Union[str, Path]] = None) -> Batch2Config:
    if path is None:
        path = Path(__file__).with_name("batch2.yaml")
    else:
        path = Path(path)
    payload: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return Batch2Config.model_validate(payload)
