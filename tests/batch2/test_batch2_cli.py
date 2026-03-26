from __future__ import annotations

import json
import sys

from plant_pipeline.cli.batch2_cli import main
from plant_pipeline.schemas.batch2 import SuspicionResult


class _FakeBatch2Service:
    def __init__(self, result):
        self.result = result

    def run_batch2(self, request):
        return self.result

    def run_batch2_folder(self, request):
        return self.result

    def close(self):
        return None


def _result() -> SuspicionResult:
    return SuspicionResult(
        image_id="img-1",
        roi_path="/tmp/roi.png",
        label="uncertain",
        suspicious=False,
        suspicious_score=0.5,
        confidence=0.25,
        lower_threshold=0.3,
        upper_threshold=0.7,
        anomaly_map_path=None,
        model_name="patchcore",
        model_version="patchcore-test-v1",
    )


def test_cli_returns_json_for_single_roi(monkeypatch, capsys):
    monkeypatch.setattr("plant_pipeline.cli.batch2_cli.load_batch2_settings", lambda path=None: object())
    monkeypatch.setattr("plant_pipeline.cli.batch2_cli.Batch2Service", lambda config: _FakeBatch2Service(_result()))
    monkeypatch.setattr(sys, "argv", ["batch2", "--roi", "/tmp/roi.png", "--image-id", "img-1"])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["image_id"] == "img-1"


def test_cli_requires_roi_or_folder(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["batch2"])
    try:
        main()
    except SystemExit as exc:
        assert "Either --roi or --folder is required." in str(exc)
