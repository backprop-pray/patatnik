from __future__ import annotations

from typing import Protocol

from plant_pipeline.schemas.batch2 import Batch2FolderRequest, Batch2FolderResult, Batch2Request, SuspicionResult


class AnomalyBackend(Protocol):
    name: str
    model_name: str
    model_version: str

    def load(self) -> None:
        ...

    def predict(self, request: Batch2Request) -> SuspicionResult:
        ...

    def predict_folder(self, request: Batch2FolderRequest) -> Batch2FolderResult:
        ...

    def close(self) -> None:
        ...
