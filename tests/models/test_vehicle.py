"""
Unit tests for road_accident_detection.models.vehicle.

These tests verify the Vehicle domain model, ensuring correct
initialization, property behavior, serialization, copying,
and object independence.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from road_accident_detection.models.detection import Detection
from road_accident_detection.models.vehicle import Vehicle


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
        tracker_id=1,
    )


def create_vehicle() -> Vehicle:
    """
    Create a reusable Vehicle instance.

    Returns:
        Vehicle object.
    """

    return Vehicle(
        tracker_id=1,
        detection=create_detection(),
    )


def test_vehicle_initialization() -> None:
    """
    Verify Vehicle initialization.
    """

    vehicle = create_vehicle()

    assert vehicle.tracker_id == 1
    assert vehicle.detection.class_name == "car"


def test_vehicle_class_name() -> None:
    """
    Verify vehicle class name.
    """

    vehicle = create_vehicle()

    assert vehicle.class_name == "car"


def test_vehicle_confidence() -> None:
    """
    Verify vehicle confidence.
    """

    vehicle = create_vehicle()

    assert vehicle.confidence == 0.95


def test_vehicle_bounding_box() -> None:
    """
    Verify vehicle bounding box.
    """

    vehicle = create_vehicle()

    assert vehicle.bounding_box == (
        100.0,
        120.0,
        300.0,
        320.0,
    )


def test_vehicle_center() -> None:
    """
    Verify vehicle center point.
    """

    vehicle = create_vehicle()

    assert vehicle.center == (
        200.0,
        220.0,
    )


def test_vehicle_area() -> None:
    """
    Verify vehicle bounding-box area.
    """

    vehicle = create_vehicle()

    assert vehicle.area == 40000.0


def test_vehicle_to_dict() -> None:
    """
    Verify dictionary serialization.
    """

    vehicle = create_vehicle()

    data = vehicle.to_dict()

    assert data["tracker_id"] == 1
    assert data["class_name"] == "car"
    assert data["confidence"] == 0.95


def test_vehicle_copy() -> None:
    """
    Verify copy behavior.
    """

    vehicle = create_vehicle()

    copied = vehicle.copy()

    assert copied is not vehicle
    assert copied == vehicle


def test_vehicle_repr() -> None:
    """
    Verify string representation.
    """

    vehicle = create_vehicle()

    representation = repr(vehicle)

    assert isinstance(representation, str)
    assert "Vehicle" in representation


def test_multiple_vehicle_instances_are_independent() -> None:
    """
    Verify Vehicle instances are independent.
    """

    first = create_vehicle()

    second = Vehicle(
        tracker_id=2,
        detection=Detection(
            class_id=7,
            class_name="truck",
            confidence=0.90,
            bounding_box=(50.0, 60.0, 250.0, 260.0),
            tracker_id=2,
        ),
    )

    assert first is not second
    assert first.tracker_id != second.tracker_id
    assert first.class_name != second.class_name


def test_vehicle_tracker_id() -> None:
    """
    Verify tracker ID.
    """

    vehicle = create_vehicle()

    assert vehicle.tracker_id > 0


def test_vehicle_detection_reference() -> None:
    """
    Verify detection reference.
    """

    detection = create_detection()

    vehicle = Vehicle(
        tracker_id=99,
        detection=detection,
    )

    assert vehicle.detection is detection


def test_vehicle_positive_area() -> None:
    """
    Verify vehicle area is positive.
    """

    vehicle = create_vehicle()

    assert vehicle.area > 0


def test_vehicle_positive_dimensions() -> None:
    """
    Verify vehicle dimensions.
    """

    vehicle = create_vehicle()

    assert vehicle.width > 0
    assert vehicle.height > 0