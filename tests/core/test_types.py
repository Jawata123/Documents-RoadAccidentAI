"""
Unit tests for road_accident_detection.core.types.

These tests verify the project's shared type aliases and ensure that the
defined types behave correctly when used throughout the application.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from road_accident_detection.core.types import (
    BoundingBox,
    Confidence,
    FrameIndex,
    ObjectID,
    PathLike,
    Point2D,
    Size2D,
)


def test_frame_index_type() -> None:
    """
    Verify FrameIndex behaves as an integer.
    """

    frame: FrameIndex = 100

    assert isinstance(frame, int)
    assert frame == 100


def test_object_id_type() -> None:
    """
    Verify ObjectID behaves as an integer.
    """

    object_id: ObjectID = 7

    assert isinstance(object_id, int)
    assert object_id == 7


def test_confidence_type() -> None:
    """
    Verify Confidence behaves as a float.
    """

    confidence: Confidence = 0.95

    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_point2d_type() -> None:
    """
    Verify Point2D structure.
    """

    point: Point2D = (125.5, 240.75)

    assert isinstance(point, tuple)
    assert len(point) == 2

    assert isinstance(point[0], float)
    assert isinstance(point[1], float)


def test_size2d_type() -> None:
    """
    Verify Size2D structure.
    """

    size: Size2D = (1920, 1080)

    assert isinstance(size, tuple)
    assert len(size) == 2

    assert size[0] > 0
    assert size[1] > 0


def test_bounding_box_type() -> None:
    """
    Verify BoundingBox structure.
    """

    bbox: BoundingBox = (
        10.0,
        20.0,
        110.0,
        220.0,
    )

    assert isinstance(bbox, tuple)
    assert len(bbox) == 4

    x_min, y_min, x_max, y_max = bbox

    assert x_min < x_max
    assert y_min < y_max


def test_pathlike_accepts_path() -> None:
    """
    Verify PathLike accepts pathlib.Path.
    """

    path: PathLike = Path("models/yolo.pt")

    assert isinstance(path, Path)


def test_pathlike_accepts_string() -> None:
    """
    Verify PathLike accepts strings.
    """

    path: PathLike = "models/yolo.pt"

    assert isinstance(path, str)


def test_numpy_image_type() -> None:
    """
    Verify NumPy image compatibility.
    """

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    assert image.shape == (480, 640, 3)
    assert image.dtype == np.uint8


def test_point_coordinates() -> None:
    """
    Verify point coordinates remain numeric.
    """

    point: Point2D = (10.5, 30.2)

    assert all(
        isinstance(value, float)
        for value in point
    )


def test_bounding_box_area() -> None:
    """
    Verify bounding box dimensions.
    """

    bbox: BoundingBox = (
        0.0,
        0.0,
        200.0,
        100.0,
    )

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    assert width == 200.0
    assert height == 100.0
    assert width * height == 20000.0


def test_size_values_positive() -> None:
    """
    Verify Size2D values are positive.
    """

    size: Size2D = (1280, 720)

    assert size[0] > 0
    assert size[1] > 0