"""
Unit tests for road_accident_detection.vision.detector.result.

These tests verify the DetectionResult model used by detector
implementations to return normalized inference results.

The DetectionResult class is the standardized output of every detector,
allowing the rest of the pipeline to remain independent of the underlying
computer vision backend.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import numpy as np

from road_accident_detection.models.detection import Detection
from road_accident_detection.vision.detector.result import DetectionResult


def create_detection() -> Detection:
    """
    Create a reusable Detection instance.

    Returns:
        Detection object.
    """

    return Detection(
        class_id=2,
        class_name="car",
        confidence=0.95,
        bounding_box=(
            100.0,
            120.0,
            300.0,
            320.0,
        ),
        tracker_id=1,
    )


def create_image() -> np.ndarray:
    """
    Create a reusable RGB image.

    Returns:
        RGB image.
    """

    return np.zeros(
        (
            480,
            640,
            3,
        ),
        dtype=np.uint8,
    )


def test_detection_result_initialization() -> None:
    """
    Verify DetectionResult initialization.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[create_detection()],
        inference_time=12.5,
    )

    assert len(result.detections) == 1
    assert result.inference_time == 12.5
    assert result.frame.shape == (
        480,
        640,
        3,
    )


def test_detection_result_empty() -> None:
    """
    Verify empty detection result.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[],
        inference_time=8.0,
    )

    assert len(result.detections) == 0


def test_detection_result_contains_detection() -> None:
    """
    Verify stored detection.
    """

    detection = create_detection()

    result = DetectionResult(
        frame=create_image(),
        detections=[detection],
        inference_time=10.0,
    )

    assert result.detections[0] is detection


def test_detection_result_detection_count() -> None:
    """
    Verify detection count.
    """

    detections = [
        create_detection(),
        create_detection(),
    ]

    result = DetectionResult(
        frame=create_image(),
        detections=detections,
        inference_time=9.5,
    )

    assert len(result.detections) == 2


def test_detection_result_frame_dimensions() -> None:
    """
    Verify frame dimensions.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[],
        inference_time=5.0,
    )

    assert result.frame.shape == (
        480,
        640,
        3,
    )


def test_detection_result_inference_time_positive() -> None:
    """
    Verify inference time is positive.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[],
        inference_time=15.3,
    )

    assert result.inference_time > 0.0


def test_detection_result_to_dict() -> None:
    """
    Verify dictionary serialization.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[create_detection()],
        inference_time=6.2,
    )

    data = result.to_dict()

    assert data["inference_time"] == 6.2
    assert len(data["detections"]) == 1


def test_detection_result_copy() -> None:
    """
    Verify copy behavior.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[create_detection()],
        inference_time=4.5,
    )

    copied = result.copy()

    assert copied is not result
    assert copied == result


def test_detection_result_repr() -> None:
    """
    Verify string representation.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[],
        inference_time=3.1,
    )

    representation = repr(result)

    assert isinstance(representation, str)
    assert "DetectionResult" in representation


def test_detection_result_frame_dtype() -> None:
    """
    Verify frame dtype.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[],
        inference_time=5.0,
    )

    assert result.frame.dtype == np.uint8


def test_multiple_detection_results_are_independent() -> None:
    """
    Verify independent DetectionResult instances.
    """

    first = DetectionResult(
        frame=create_image(),
        detections=[],
        inference_time=5.0,
    )

    second = DetectionResult(
        frame=create_image(),
        detections=[create_detection()],
        inference_time=8.0,
    )

    assert first is not second
    assert len(first.detections) != len(second.detections)


def test_detection_result_image_is_numpy_array() -> None:
    """
    Verify stored frame is a NumPy array.
    """

    result = DetectionResult(
        frame=create_image(),
        detections=[],
        inference_time=2.0,
    )

    assert isinstance(result.frame, np.ndarray)