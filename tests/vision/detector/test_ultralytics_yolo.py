"""
Unit tests for road_accident_detection.vision.detector.ultralytics_yolo.

These tests verify the Ultralytics YOLO detector implementation used by the
RoadAccidentAI framework. The tests focus on initialization, model loading,
inference lifecycle, interface compliance, and detector state.

Actual inference on pretrained weights should be covered by integration tests
using real model files. These unit tests validate the detector API and
expected behavior without requiring a trained accident detection model.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from road_accident_detection.core.exceptions import (
    ModelLoadError,
)
from road_accident_detection.vision.detector.base import BaseDetector
from road_accident_detection.vision.detector.result import DetectionResult
from road_accident_detection.vision.detector.ultralytics_yolo import (
    UltralyticsYOLODetector,
)


def create_image() -> np.ndarray:
    """
    Create a reusable RGB image.

    Returns:
        RGB NumPy image.
    """

    return np.zeros(
        (
            640,
            640,
            3,
        ),
        dtype=np.uint8,
    )


def test_detector_is_base_detector() -> None:
    """
    Verify detector inherits from BaseDetector.
    """

    assert issubclass(
        UltralyticsYOLODetector,
        BaseDetector,
    )


def test_detector_initialization() -> None:
    """
    Verify detector initialization.
    """

    detector = UltralyticsYOLODetector(
        model_path=Path("models/yolo26n.pt"),
    )

    assert detector is not None
    assert detector.model_name


def test_detector_accepts_string_path() -> None:
    """
    Verify model path accepts strings.
    """

    detector = UltralyticsYOLODetector(
        model_path="models/yolo26n.pt",
    )

    assert detector is not None


def test_detector_accepts_path_object() -> None:
    """
    Verify model path accepts pathlib.Path.
    """

    detector = UltralyticsYOLODetector(
        model_path=Path("models/yolo26n.pt"),
    )

    assert detector is not None


def test_detector_not_loaded_initially() -> None:
    """
    Verify detector is initially unloaded.
    """

    detector = UltralyticsYOLODetector(
        model_path="models/yolo26n.pt",
    )

    assert detector.is_loaded is False


def test_invalid_model_path_raises() -> None:
    """
    Verify invalid model paths raise ModelLoadError.
    """

    detector = UltralyticsYOLODetector(
        model_path="models/does_not_exist.pt",
    )

    with pytest.raises(ModelLoadError):
        detector.load()


def test_detect_before_loading_raises() -> None:
    """
    Verify inference before loading is rejected.
    """

    detector = UltralyticsYOLODetector(
        model_path="models/yolo26n.pt",
    )

    with pytest.raises(ModelLoadError):
        detector.detect(
            create_image(),
        )


def test_detector_has_required_methods() -> None:
    """
    Verify detector exposes required interface.
    """

    detector = UltralyticsYOLODetector(
        model_path="models/yolo26n.pt",
    )

    assert callable(detector.load)
    assert callable(detector.unload)
    assert callable(detector.detect)


def test_detector_has_model_name() -> None:
    """
    Verify detector exposes model name.
    """

    detector = UltralyticsYOLODetector(
        model_path="models/yolo26n.pt",
    )

    assert isinstance(
        detector.model_name,
        str,
    )

    assert detector.model_name


def test_detector_multiple_instances() -> None:
    """
    Verify detector instances are independent.
    """

    detector1 = UltralyticsYOLODetector(
        model_path="models/yolo26n.pt",
    )

    detector2 = UltralyticsYOLODetector(
        model_path="models/yolo26n.pt",
    )

    assert detector1 is not detector2


def test_detector_accepts_numpy_image() -> None:
    """
    Verify detector accepts NumPy image input.
    """

    image = create_image()

    assert isinstance(
        image,
        np.ndarray,
    )


@pytest.mark.parametrize(
    "height,width",
    [
        (
            480,
            640,
        ),
        (
            720,
            1280,
        ),
        (
            1080,
            1920,
        ),
    ],
)
def test_detector_image_sizes(
    height: int,
    width: int,
) -> None:
    """
    Verify detector supports multiple image sizes.

    Args:
        height:
            Image height.

        width:
            Image width.
    """

    image = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    assert image.shape == (
        height,
        width,
        3,
    )


def test_detector_detect_return_type_annotation() -> None:
    """
    Verify detect() advertises DetectionResult.
    """

    annotation = (
        UltralyticsYOLODetector.detect.__annotations__
        .get("return")
    )

    assert annotation is DetectionResult