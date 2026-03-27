from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from plant_pipeline.detect.backends.ultralytics_backend import UltralyticsLeafBackend


class _FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeYOLO:
    def __init__(self, path):
        self.path = path

    def predict(self, image, save=False, verbose=False, device=None):
        boxes = SimpleNamespace(
            cls=_FakeArray([0, 1]),
            conf=_FakeArray([0.8, 0.4]),
            xyxy=_FakeArray([[10, 20, 40, 60], [100, 110, 140, 160]]),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "leaf", 1: "plant"})]


def test_converts_ultralytics_output_to_detection_boxes(monkeypatch):
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace(YOLO=_FakeYOLO))
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    backend = UltralyticsLeafBackend("/tmp/yolo11x_leaf.pt")
    detections = backend.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(detections) == 2
    assert detections[0].label == "leaf"
    assert detections[0].bbox.x_min == 10


def test_normalizes_species_labels_to_leaf(monkeypatch):
    class _SpeciesYOLO:
        def __init__(self, path):
            self.path = path

        def predict(self, image, save=False, verbose=False, device=None):
            boxes = SimpleNamespace(
                cls=_FakeArray([0, 1]),
                conf=_FakeArray([0.8, 0.4]),
                xyxy=_FakeArray([[10, 20, 40, 60], [100, 110, 140, 160]]),
            )
            return [SimpleNamespace(boxes=boxes, names={0: "corn", 1: "weed"})]

    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace(YOLO=_SpeciesYOLO))
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    backend = UltralyticsLeafBackend("/tmp/foduu_plant_leaf_yolov8s_best.pt")
    detections = backend.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert [item.label for item in detections] == ["leaf", "leaf"]


def test_reports_agpl_license_tag():
    backend = UltralyticsLeafBackend("/tmp/yolo11x_leaf.pt")
    assert backend.license_tag == "AGPL-3.0-or-commercial"
