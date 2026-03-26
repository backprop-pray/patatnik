from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


SuspicionLabel = Literal["normal", "suspicious", "uncertain"]


class Batch2Request(BaseModel):
    image_id: str
    roi_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Batch2FolderRequest(BaseModel):
    input_dir: str
    glob_pattern: str = "*.png"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuspicionResult(BaseModel):
    image_id: str
    roi_path: str
    label: SuspicionLabel
    suspicious: bool
    suspicious_score: float
    confidence: float
    lower_threshold: float
    upper_threshold: float
    anomaly_map_path: Optional[str] = None
    model_name: str
    model_version: str
    debug: dict[str, Any] = Field(default_factory=dict)


class Batch2FolderResult(BaseModel):
    results: list[SuspicionResult] = Field(default_factory=list)
    processed_count: int = 0
    failed_count: int = 0
    debug: dict[str, Any] = Field(default_factory=dict)


class ThresholdBundle(BaseModel):
    lower_threshold: float
    upper_threshold: float
    normal_percentile: float
    suspicious_percentile: float
    calibration_dataset_version: str
    score_summary: dict[str, float] = Field(default_factory=dict)


class Batch2ModelBundle(BaseModel):
    bundle_dir: str
    model_name: str
    model_version: str
    checkpoint_path: str
    metadata_path: str
    thresholds: ThresholdBundle


class Batch2ErrorResult(BaseModel):
    image_id: str
    roi_path: str
    error_code: str
    error_message: str
    debug: dict[str, Any] = Field(default_factory=dict)
