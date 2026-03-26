from __future__ import annotations

from plant_pipeline.anomaly.calibration import calibrate_thresholds
from plant_pipeline.config.settings import Batch2ThresholdSettings


def test_calibration_computes_valid_thresholds():
    bundle = calibrate_thresholds(
        [0.10, 0.12, 0.18, 0.22],
        [0.72, 0.81, 0.88],
        Batch2ThresholdSettings(),
        dataset_version="dataset-v1",
    )
    assert bundle.upper_threshold > bundle.lower_threshold


def test_calibration_supports_normal_only_fallback():
    bundle = calibrate_thresholds(
        [0.10, 0.12, 0.18, 0.22],
        [],
        Batch2ThresholdSettings(),
        dataset_version="dataset-v1",
    )
    assert bundle.upper_threshold > bundle.lower_threshold
