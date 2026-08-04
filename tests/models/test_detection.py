"""
Unit tests for road_accident_detection.models.detection.

These tests verify the Detection domain model, ensuring correct
initialization, computed properties, serialization, copying,
and object independence.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from road_accident_detection.models.detection import Detection


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
        bounding_box=(100.0, 120.0, 300.0, 320.0),
        tracker_id=5,
    )


def test_detection_initialization() -> None:
    """
    Verify Detection initialization.
    """

    detection = create_detection()

    assert detection.class_id == 2
    assert detection.class_name == "car"
    assert detection.confidence == 0.95
    assert detection.bounding_box == (
        100.0,
        120.0,
        300.0,
        320.0,
    )
    assert detection.tracker_id == 5


def test_detection_dimensions() -> None:
    """
    Verify bounding-box dimensions.
    """

    detection = create_detection()

    assert detection.width == 200.0
    assert detection.height == 200.0


def test_detection_center() -> None:
    """
    Verify center point calculation.
    """

    detection = create_detection()

    assert detection.center == (
        200.0,
        220.0,
    )


def test_detection_area() -> None:
    """
    Verify bounding-box area.
    """

    detection = create_detection()

    assert detection.area == 40000.0


def test_detection_is_tracked() -> None:
    """
    Verify tracked detection.
    """

    detection = create_detection()

    assert detection.is_tracked is True


def test_untracked_detection() -> None:
    """
    Verify detection without tracker ID.
    """

    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.88,
        bounding_box=(10.0, 20.0, 50.0, 100.0),
    )

    assert detection.is_tracked is False


def test_detection_to_dict() -> None:
    """
    Verify dictionary serialization.
    """

    detection = create_detection()

    data = detection.to_dict()

    assert data["class_id"] == 2
    assert data["class_name"] == "car"
    assert data["confidence"] == 0.95
    assert data["tracker_id"] == 5
    assert data["bounding_box"] == (
        100.0,
        120.0,
        300.0,
        320.0,
    )


def test_detection_copy() -> None:
    """
    Verify copy behavior.
    """

    detection = create_detection()

    copied = detection.copy()

    assert copied is not detection
    assert copied == detection


def test_detection_repr() -> None:
    """
    Verify string representation.
    """

    detection = create_detection()

    representation = repr(detection)

    assert isinstance(representation, str)
    assert "Detection" in representation


def test_detection_confidence_range() -> None:
    """
    Verify confidence value.
    """

    detection = create_detection()

    assert 0.0 <= detection.confidence <= 1.0


def test_multiple_detections_are_independent() -> None:
    """
    Verify Detection instances are independent.
    """

    first = create_detection()

    second = Detection(
        class_id=7,
        class_name="truck",
        confidence=0.90,
        bounding_box=(50.0, 60.0, 250.0, 260.0),
    )

    assert first is not second
    assert first.class_name != second.class_name


def test_detection_bbox_coordinates() -> None:
    """
    Verify bounding-box coordinates.
    """

    detection = create_detection()

    x1, y1, x2, y2 = detection.bounding_box

    assert x1 < x2
    assert y1 < y2


def test_detection_positive_area() -> None:
    """
    Verify detection area is positive.
    """

    detection = create_detection()

    assert detection.area > 0


def test_detection_width_height_positive() -> None:
    """
    Verify width and height are positive.
    """

    detection = create_detection()

    assert detection.width > 0
    assert detection.height > 0