from __future__ import annotations

import argparse
import json
import logging

from plant_pipeline.config.settings import load_batch2_settings
from plant_pipeline.schemas.batch2 import Batch2FolderRequest, Batch2Request
from plant_pipeline.services.batch2_service import Batch2Service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch 2 ROI anomaly scoring")
    parser.add_argument("--roi", default=None, help="Path to a single ROI image")
    parser.add_argument("--image-id", default=None, help="Image id for single ROI inference")
    parser.add_argument("--folder", default=None, help="Path to ROI folder for batch inference")
    parser.add_argument("--glob", default=None, help="Glob pattern for folder inference")
    parser.add_argument("--config", default=None, help="Path to batch2 YAML config")
    parser.add_argument("--write-anomaly-map", choices=["true", "false"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.roi and not args.folder:
        raise SystemExit("Either --roi or --folder is required.")
    config = load_batch2_settings(args.config)
    if args.write_anomaly_map is not None:
        config.batch2.write_anomaly_map = args.write_anomaly_map == "true"
    logging.basicConfig(level=logging.INFO)
    service = Batch2Service(config)
    try:
        if args.folder:
            request = Batch2FolderRequest(
                input_dir=args.folder,
                glob_pattern=args.glob or config.batch2.roi_glob_pattern,
            )
            payload = service.run_batch2_folder(request)
        else:
            image_id = args.image_id or __import__("pathlib").Path(args.roi).stem
            payload = service.run_batch2(Batch2Request(image_id=image_id, roi_path=args.roi))
        print(json.dumps(payload.model_dump(mode="json"), indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
