"""
Unit tests for road_accident_detection.vision.detector.factory.

These tests verify the detector factory responsible for creating detector
instances. The factory provides a single entry point for constructing
supported detector implementations while hiding backend-specific creation
details from the rest of the application.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from road_accident_detection.core.exceptions import ConfigurationError
from road_accident_detection.vision.detector.base import BaseDetector
from road_accident_detection.vision.detector.factory import (
    DetectorFactory,
)


def test_factory_is_instantiable() -> None:
    """
    Verify DetectorFactory can be instantiated.
    """

    factory = DetectorFactory()

    assert isinstance(factory, DetectorFactory)


def test_factory_create_ultralytics_detector() -> None:
    """
    Verify creation of the Ultralytics detector.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    assert isinstance(detector, BaseDetector)


def test_factory_case_insensitive_detector_name() -> None:
    """
    Verify detector names are case-insensitive.
    """

    detector = DetectorFactory.create(
        detector_type="UlTrAlYtIcS",
        model_path=Path("models/yolo26n.pt"),
    )

    assert isinstance(detector, BaseDetector)


@pytest.mark.parametrize(
    "detector_name",
    [
        "unknown",
        "invalid",
        "tensorflow",
        "opencv",
        "",
    ],
)
def test_factory_invalid_detector_name(
    detector_name: str,
) -> None:
    """
    Verify unsupported detector names raise ConfigurationError.

    Args:
        detector_name:
            Detector backend name.
    """

    with pytest.raises(ConfigurationError):
        DetectorFactory.create(
            detector_type=detector_name,
            model_path=Path("models/yolo26n.pt"),
        )


def test_factory_returns_new_instance() -> None:
    """
    Verify each factory call returns a new detector instance.
    """

    detector1 = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    detector2 = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    assert detector1 is not detector2


def test_factory_detector_has_model_name() -> None:
    """
    Verify detector exposes a model name.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    assert isinstance(detector.model_name, str)
    assert detector.model_name


def test_factory_detector_has_loaded_state() -> None:
    """
    Verify detector exposes loading state.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    assert isinstance(detector.is_loaded, bool)


def test_factory_returns_base_detector() -> None:
    """
    Verify factory returns BaseDetector implementations.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    assert issubclass(
        detector.__class__,
        BaseDetector,
    )


def test_factory_accepts_path_object() -> None:
    """
    Verify pathlib.Path is accepted.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    assert detector is not None


def test_factory_accepts_string_path() -> None:
    """
    Verify string paths are accepted.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path="models/yolo26n.pt",
    )

    assert detector is not None


def test_factory_multiple_instances_independent() -> None:
    """
    Verify detectors returned by the factory are independent.
    """

    detectors = [
        DetectorFactory.create(
            detector_type="ultralytics",
            model_path=Path("models/yolo26n.pt"),
        )
        for _ in range(3)
    ]

    assert detectors[0] is not detectors[1]
    assert detectors[1] is not detectors[2]
    assert detectors[0] is not detectors[2]


def test_factory_create_returns_correct_type() -> None:
    """
    Verify create() returns an object implementing BaseDetector.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    assert hasattr(detector, "detect")
    assert hasattr(detector, "load")
    assert hasattr(detector, "unload")