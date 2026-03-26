from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from plant_pipeline.config.settings import load_batch1_settings, load_batch2_settings
from plant_pipeline.schemas.batch1 import Batch1Request
from plant_pipeline.schemas.batch2 import Batch2Request
from plant_pipeline.services.batch1_service import Batch1Service
from plant_pipeline.services.batch2_service import Batch2Service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full plant pipeline on one image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--batch1-config", default=None, help="Path to the batch1 YAML config")
    parser.add_argument("--batch2-config", default=None, help="Path to the batch2 YAML config")
    parser.add_argument("--image-id", default=None, help="Optional image id override")
    parser.add_argument("--mission-id", default=None)
    parser.add_argument("--row-id", default=None)
    parser.add_argument("--section-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_path = str(Path(args.image).expanduser())

    batch1_config_path = args.batch1_config
    if batch1_config_path is None:
        default_local = Path(__file__).resolve().parents[1] / "config" / "batch1.ultralytics.local.yaml"
        if default_local.exists():
            batch1_config_path = str(default_local)

    logging.basicConfig(level=logging.INFO)
    batch1_service = Batch1Service(load_batch1_settings(batch1_config_path))
    try:
        batch1_result = batch1_service.run(
            Batch1Request(
                image_path=image_path,
                image_id=args.image_id,
                mission_id=args.mission_id,
                row_id=args.row_id,
                section_id=args.section_id,
            )
        )
    finally:
        batch1_service.close()

    payload: dict[str, object] = {
        "image_path": image_path,
        "batch1_config": batch1_config_path,
        "batch2_config": args.batch2_config,
        "batch1": batch1_result.model_dump(mode="json"),
        "batch2": None,
    }

    roi_path = None
    if batch1_result.localization is not None:
        roi_path = batch1_result.localization.roi_path
    if roi_path is None:
        roi_path = batch1_result.artifacts.get("roi_path")

    if batch1_result.valid and batch1_result.contains_plant and roi_path:
        batch2_service = Batch2Service(load_batch2_settings(args.batch2_config))
        try:
            batch2_result = batch2_service.run_batch2(
                Batch2Request(
                    image_id=batch1_result.image_id,
                    roi_path=roi_path,
                    metadata={
                        "source_image_path": image_path,
                        "batch1_contains_plant": batch1_result.contains_plant,
                    },
                )
            )
            payload["batch2"] = batch2_result.model_dump(mode="json")
        finally:
            batch2_service.close()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
