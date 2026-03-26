from __future__ import annotations

from fastapi import FastAPI

from plant_pipeline.config.settings import load_batch2_settings
from plant_pipeline.schemas.batch2 import Batch2FolderRequest, Batch2Request
from plant_pipeline.services.batch2_service import Batch2Service


def create_app(config_path: str | None = None) -> FastAPI:
    config = load_batch2_settings(config_path)
    service = Batch2Service(config)
    app = FastAPI(title="Plant Pipeline Batch2 API")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        service.close()

    @app.post("/batch2/infer")
    def infer(request: Batch2Request):
        return service.run_batch2(request)

    @app.post("/batch2/infer-folder")
    def infer_folder(request: Batch2FolderRequest):
        return service.run_batch2_folder(request)

    return app
