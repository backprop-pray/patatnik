from __future__ import annotations

from plant_pipeline.roi.select import (
    ensure_minimum_roi_size,
    expand_and_clip_bbox,
    gather_single_detection_context,
    score_clusters,
    select_best_cluster,
)
from plant_pipeline.schemas.batch1 import BoundingBox, DetectionBox


def test_prefers_central_dense_cluster(batch1_config, dense_leaf_detections):
    edge_cluster = [
        DetectionBox(bbox=BoundingBox(x_min=10, y_min=10, x_max=80, y_max=80), confidence=0.95, label="leaf"),
    ]
    clusters = score_clusters([dense_leaf_detections, edge_cluster], (640, 640, 3), batch1_config.cluster)
    selected = select_best_cluster(clusters, batch1_config.cluster)
    assert selected is not None
    assert selected.cluster_id == 0


def test_rejects_cluster_below_area_threshold(batch1_config):
    batch1_config.cluster.min_cluster_area_ratio = 0.1
    small_cluster = [[DetectionBox(bbox=BoundingBox(x_min=10, y_min=10, x_max=30, y_max=30), confidence=0.7, label="leaf")]]
    clusters = score_clusters(small_cluster, (640, 640, 3), batch1_config.cluster)
    assert select_best_cluster(clusters, batch1_config.cluster) is None


def test_expands_and_clips_bbox_at_border(batch1_config):
    bbox = BoundingBox(x_min=5, y_min=5, x_max=40, y_max=50)
    expanded = expand_and_clip_bbox(bbox, (100, 100, 3), batch1_config.cluster.bbox_expand_ratio)
    assert 0 <= expanded.x_min < bbox.x_min
    assert 0 <= expanded.y_min < bbox.y_min
    assert expanded.x_max <= 100
    assert expanded.y_max <= 100


def test_penalizes_oversized_border_touching_cluster(batch1_config):
    oversized_cluster = [
        DetectionBox(bbox=BoundingBox(x_min=0, y_min=0, x_max=640, y_max=330), confidence=0.95, label="leaf"),
        DetectionBox(bbox=BoundingBox(x_min=0, y_min=300, x_max=640, y_max=640), confidence=0.95, label="leaf"),
    ]
    central_cluster = [
        DetectionBox(bbox=BoundingBox(x_min=230, y_min=160, x_max=370, y_max=320), confidence=0.68, label="leaf"),
        DetectionBox(bbox=BoundingBox(x_min=290, y_min=200, x_max=430, y_max=360), confidence=0.66, label="leaf"),
    ]
    clusters = score_clusters([oversized_cluster, central_cluster], (640, 640, 3), batch1_config.cluster)
    selected = select_best_cluster(clusters, batch1_config.cluster)
    assert selected is not None
    assert selected.cluster_id == 1


def test_gathers_nearby_context_for_single_detection(batch1_config):
    selected_cluster = score_clusters(
        [[DetectionBox(bbox=BoundingBox(x_min=500, y_min=300, x_max=560, y_max=420), confidence=0.9, label="leaf")]],
        (640, 640, 3),
        batch1_config.cluster,
    )[0]
    detections = [
        DetectionBox(bbox=BoundingBox(x_min=500, y_min=300, x_max=560, y_max=420), confidence=0.9, label="leaf"),
        DetectionBox(bbox=BoundingBox(x_min=430, y_min=280, x_max=490, y_max=390), confidence=0.4, label="leaf"),
    ]
    gathered = gather_single_detection_context(selected_cluster, detections, (640, 640, 3), batch1_config.cluster)
    assert gathered.x_min == 430
    assert gathered.x_max == 560


def test_ensures_minimum_roi_size(batch1_config):
    bbox = BoundingBox(x_min=500, y_min=300, x_max=560, y_max=420)
    resized = ensure_minimum_roi_size(
        bbox,
        (640, 640, 3),
        batch1_config.cluster.min_final_roi_width_ratio,
        batch1_config.cluster.min_final_roi_height_ratio,
    )
    assert resized.width >= int(640 * batch1_config.cluster.min_final_roi_width_ratio)
    assert resized.height >= int(640 * batch1_config.cluster.min_final_roi_height_ratio)
