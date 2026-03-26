from __future__ import annotations

from pathlib import Path
from typing import Optional

from plant_pipeline.anomaly.backends.patchcore_backend import PatchCoreBackend
from plant_pipeline.anomaly.base import AnomalyBackend
from plant_pipeline.config.settings import Batch2Config, load_batch2_settings
from plant_pipeline.schemas.batch2 import Batch2FolderRequest, Batch2FolderResult, Batch2Request, SuspicionResult


class Batch2Service:
    def __init__(self, config: Batch2Config, backend: Optional[AnomalyBackend] = None) -> None:
        self.config = config
        self.backend = backend or PatchCoreBackend(config)
        self.backend.load()

    def run_batch2(self, request: Batch2Request) -> SuspicionResult:
        roi_path = Path(request.roi_path)
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI not found: {request.roi_path}")
        return self.backend.predict(request)

    def run_batch2_folder(self, request: Batch2FolderRequest) -> Batch2FolderResult:
        input_dir = Path(request.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"ROI folder not found: {request.input_dir}")
        return self.backend.predict_folder(request)

    def close(self) -> None:
        self.backend.close()


def build_batch2_service(config_path: str | None = None) -> Batch2Service:
    return Batch2Service(load_batch2_settings(config_path))
