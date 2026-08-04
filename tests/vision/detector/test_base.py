"""
Unit tests for road_accident_detection.vision.detector.base.

These tests verify the abstract detector interface used throughout the
RoadAccidentAI framework. Every detector backend (Ultralytics YOLO,
future RT-DETR, Grounding DINO, etc.) must satisfy this interface.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from abc import ABC

import numpy as np
import pytest

from road_accident_detection.models.detection import Detection
from road_accident_detection.vision.detector.base import BaseDetector
from road_accident_detection.vision.detector.result import DetectionResult


class DummyDetector(BaseDetector):
    """
    Minimal detector implementation for interface testing.
    """

    @property
    def model_name(self) -> str:
        """Return detector name."""
        return "dummy"

    @property
    def is_loaded(self) -> bool:
        """Return loading state."""
        return True

    def load(self) -> None:
        """Load detector."""
        return None

    def unload(self) -> None:
        """Unload detector."""
        return None

    def detect(
        self,
        frame: np.ndarray,
    ) -> DetectionResult:
        """
        Return an empty detection result.

        Args:
            frame:
                Input RGB frame.

        Returns:
            DetectionResult.
        """

        return DetectionResult(
            frame=frame,
            detections=[],
            inference_time=1.0,
        )


def create_image() -> np.ndarray:
    """
    Create a reusable RGB image.

    Returns:
        NumPy image.
    """

    return np.zeros(
        (
            480,
            640,
            3,
        ),
        dtype=np.uint8,
    )


def test_base_detector_is_abstract() -> None:
    """
    Verify BaseDetector is abstract.
    """

    assert issubclass(
        BaseDetector,
        ABC,
    )


def test_dummy_detector_creation() -> None:
    """
    Verify detector can be instantiated.
    """

    detector = DummyDetector()

    assert isinstance(
        detector,
        BaseDetector,
    )


def test_detector_name() -> None:
    """
    Verify detector name.
    """

    detector = DummyDetector()

    assert detector.model_name == "dummy"


def test_detector_loaded_state() -> None:
    """
    Verify detector loading state.
    """

    detector = DummyDetector()

    assert detector.is_loaded is True


def test_detector_load() -> None:
    """
    Verify load method executes.
    """

    detector = DummyDetector()

    detector.load()

    assert detector.is_loaded


def test_detector_unload() -> None:
    """
    Verify unload method executes.
    """

    detector = DummyDetector()

    detector.unload()

    assert detector is not None


def test_detector_detect_returns_result() -> None:
    """
    Verify detect returns DetectionResult.
    """

    detector = DummyDetector()

    result = detector.detect(
        create_image(),
    )

    assert isinstance(
        result,
        DetectionResult,
    )


def test_detector_result_frame() -> None:
    """
    Verify returned frame.
    """

    detector = DummyDetector()

    frame = create_image()

    result = detector.detect(frame)

    assert result.frame is frame


def test_detector_result_detections() -> None:
    """
    Verify detection list.
    """

    detector = DummyDetector()

    result = detector.detect(
        create_image(),
    )

    assert isinstance(
        result.detections,
        list,
    )

    assert len(result.detections) == 0


def test_detector_result_time() -> None:
    """
    Verify inference time.
    """

    detector = DummyDetector()

    result = detector.detect(
        create_image(),
    )

    assert result.inference_time > 0.0


def test_detector_multiple_calls() -> None:
    """
    Verify detector supports repeated inference.
    """

    detector = DummyDetector()

    result1 = detector.detect(
        create_image(),
    )

    result2 = detector.detect(
        create_image(),
    )

    assert result1 is not result2


def test_detector_accepts_numpy_image() -> None:
    """
    Verify detector accepts NumPy images.
    """

    detector = DummyDetector()

    image = create_image()

    result = detector.detect(image)

    assert result.frame.shape == (
        480,
        640,
        3,
    )


def test_detector_returns_detection_list() -> None:
    """
    Verify detections are Detection objects.
    """

    detector = DummyDetector()

    result = detector.detect(
        create_image(),
    )

    assert all(
        isinstance(
            detection,
            Detection,
        )
        for detection in result.detections
    )


@pytest.mark.parametrize(
    "height,width",
    [
        (480, 640),
        (720, 1280),
        (1080, 1920),
    ],
)
def test_detector_various_image_sizes(
    height: int,
    width: int,
) -> None:
    """
    Verify detector accepts various image sizes.

    Args:
        height:
            Image height.

        width:
            Image width.
    """

    detector = DummyDetector()

    image = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    result = detector.detect(image)

    assert result.frame.shape == (
        height,
        width,
        3,
    )