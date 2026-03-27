from __future__ import annotations

import argparse
import logging
import time
import uuid
from pathlib import Path

import cv2

from plant_pipeline.config.settings import load_batch1_settings, load_batch2_settings
from plant_pipeline.schemas.batch1 import Batch1Request, BoundingBox
from plant_pipeline.schemas.batch2 import Batch2Request, SuspicionResult
from plant_pipeline.services.batch1_service import Batch1Service
from plant_pipeline.services.batch2_service import Batch2Service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full plant pipeline on a live camera feed")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--batch1-config", default=None, help="Path to the batch1 YAML config")
    parser.add_argument("--batch2-config", default=None, help="Path to the batch2 YAML config")
    parser.add_argument("--interval-frames", type=int, default=20, help="Run inference every N frames")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--window-name", default="Plant Pipeline Live", help="OpenCV window title")
    parser.add_argument(
        "--captures-dir",
        default="./data/live_camera_frames",
        help="Directory where analyzed frames are temporarily written",
    )
    return parser


def _default_batch1_config(cli_value: str | None) -> str | None:
    if cli_value is not None:
        return cli_value
    default_local = Path(__file__).resolve().parents[1] / "config" / "batch1.ultralytics.local.yaml"
    if default_local.exists():
        return str(default_local)
    return None


def _draw_bbox(frame, bbox: BoundingBox, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), color, 3)


def _put_status_line(frame, text: str, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def _label_color(label: str | None) -> tuple[int, int, int]:
    if label == "suspicious":
        return (0, 0, 255)
    if label == "uncertain":
        return (0, 215, 255)
    return (0, 200, 0)


def _run_full_pipeline_for_frame(
    frame_path: str,
    frame_id: str,
    batch1_service: Batch1Service,
    batch2_service: Batch2Service,
) -> tuple[object, SuspicionResult | None]:
    batch1_result = batch1_service.run(Batch1Request(image_path=frame_path, image_id=frame_id))
    roi_path = None
    if batch1_result.localization is not None:
        roi_path = batch1_result.localization.roi_path
    if roi_path is None:
        roi_path = batch1_result.artifacts.get("roi_path")

    batch2_result = None
    if batch1_result.valid and batch1_result.contains_plant and roi_path:
        batch2_result = batch2_service.run_batch2(
            Batch2Request(
                image_id=batch1_result.image_id,
                roi_path=roi_path,
                metadata={"source_image_path": frame_path, "batch1_contains_plant": batch1_result.contains_plant},
            )
        )
    return batch1_result, batch2_result


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    batch1_config_path = _default_batch1_config(args.batch1_config)
    batch2_config_path = args.batch2_config
    captures_dir = Path(args.captures_dir)
    captures_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not capture.isOpened():
        raise RuntimeError(f"Failed to open camera index {args.camera_index}")

    batch1_service = Batch1Service(load_batch1_settings(batch1_config_path))
    batch2_service = Batch2Service(load_batch2_settings(batch2_config_path))

    last_batch1 = None
    last_batch2 = None
    last_latency_ms = None
    last_processed_at = 0.0
    frame_idx = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read from camera")

            frame_idx += 1
            should_run = frame_idx == 1 or frame_idx % max(args.interval_frames, 1) == 0

            if should_run:
                frame_id = uuid.uuid4().hex
                frame_path = captures_dir / f"{frame_id}.jpg"
                cv2.imwrite(str(frame_path), frame)
                started = time.perf_counter()
                last_batch1, last_batch2 = _run_full_pipeline_for_frame(
                    str(frame_path), frame_id, batch1_service, batch2_service
                )
                last_latency_ms = (time.perf_counter() - started) * 1000.0
                last_processed_at = time.time()

            display = frame.copy()

            if last_batch1 is not None and last_batch1.localization and last_batch1.localization.bbox is not None:
                bbox_color = _label_color(last_batch2.label if last_batch2 is not None else None)
                _draw_bbox(display, last_batch1.localization.bbox, bbox_color)

            header_color = (255, 255, 255)
            if last_batch1 is None:
                _put_status_line(display, "Waiting for first inference...", 32, header_color)
            elif not last_batch1.valid:
                _put_status_line(display, f"Rejected: {last_batch1.reject_reason}", 32, (0, 165, 255))
            elif not last_batch1.contains_plant:
                _put_status_line(display, "No plant detected", 32, (0, 165, 255))
            elif last_batch2 is None:
                _put_status_line(display, "Plant detected, Batch 2 not run", 32, (0, 165, 255))
            else:
                _put_status_line(
                    display,
                    f"Plant detected | Batch 2: {last_batch2.label} | score={last_batch2.suspicious_score:.2f}",
                    32,
                    _label_color(last_batch2.label),
                )

            if last_batch1 is not None and last_batch1.quality is not None:
                _put_status_line(
                    display,
                    (
                        f"Blur={last_batch1.quality.diagnostics.blur_score:.1f} "
                        f"Veg={last_batch1.quality.diagnostics.vegetation_fraction or 0.0:.2f}"
                    ),
                    64,
                    header_color,
                )

            if last_latency_ms is not None:
                age_seconds = max(0.0, time.time() - last_processed_at)
                _put_status_line(
                    display,
                    f"Inference {last_latency_ms:.0f} ms | age {age_seconds:.1f}s | interval {args.interval_frames} frames",
                    96,
                    header_color,
                )

            _put_status_line(display, "Press q to quit, space to force an immediate inference", 128, header_color)
            cv2.imshow(args.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                frame_idx = 0

    finally:
        capture.release()
        cv2.destroyAllWindows()
        batch1_service.close()
        batch2_service.close()


if __name__ == "__main__":
    main()
