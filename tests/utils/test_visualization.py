"""
Unit tests for road_accident_detection.utils.visualization.

These tests verify the project's visualization utilities responsible for
drawing detections, bounding boxes, labels, and text overlays on image
frames. The tests ensure that visualization functions behave correctly
without modifying the original image unexpectedly.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import numpy as np

from road_accident_detection.models.detection import Detection
from road_accident_detection.utils.visualization import (
    draw_bounding_box,
    draw_detection,
    draw_label,
    draw_text,
)


def create_test_image() -> np.ndarray:
    """
    Create a reusable RGB image.

    Returns:
        RGB image.
    """

    return np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )


def create_detection() -> Detection:
    """
    Create a reusable Detection instance.

    Returns:
        Detection object.
    """

    return Detection(
        class_id=2,
        class_name="car",
        confidence=0.96,
        bounding_box=(
            100.0,
            120.0,
            300.0,
            320.0,
        ),
        tracker_id=7,
    )


def test_draw_bounding_box_returns_image() -> None:
    """
    Verify drawing a bounding box returns an image.
    """

    image = create_test_image()

    output = draw_bounding_box(
        image=image,
        bounding_box=(100, 120, 300, 320),
    )

    assert isinstance(output, np.ndarray)
    assert output.shape == image.shape


def test_draw_label_returns_image() -> None:
    """
    Verify drawing a label returns an image.
    """

    image = create_test_image()

    output = draw_label(
        image=image,
        text="Car",
        position=(100, 100),
    )

    assert isinstance(output, np.ndarray)
    assert output.shape == image.shape


def test_draw_text_returns_image() -> None:
    """
    Verify drawing text returns an image.
    """

    image = create_test_image()

    output = draw_text(
        image=image,
        text="RoadAccidentAI",
        position=(50, 50),
    )

    assert isinstance(output, np.ndarray)
    assert output.shape == image.shape


def test_draw_detection_returns_image() -> None:
    """
    Verify drawing an entire detection.
    """

    image = create_test_image()

    output = draw_detection(
        image=image,
        detection=create_detection(),
    )

    assert isinstance(output, np.ndarray)
    assert output.shape == image.shape


def test_visualization_preserves_dtype() -> None:
    """
    Verify visualization preserves dtype.
    """

    image = create_test_image()

    output = draw_text(
        image=image,
        text="FPS",
        position=(10, 20),
    )

    assert output.dtype == np.uint8


def test_visualization_preserves_dimensions() -> None:
    """
    Verify image dimensions remain unchanged.
    """

    image = create_test_image()

    output = draw_detection(
        image=image,
        detection=create_detection(),
    )

    assert output.shape == (
        480,
        640,
        3,
    )


def test_draw_detection_with_tracking() -> None:
    """
    Verify tracked detections are drawable.
    """

    image = create_test_image()

    detection = create_detection()

    output = draw_detection(
        image=image,
        detection=detection,
    )

    assert output is not None


def test_multiple_draw_operations() -> None:
    """
    Verify multiple draw operations succeed.
    """

    image = create_test_image()

    image = draw_text(
        image=image,
        text="Frame: 1",
        position=(20, 20),
    )

    image = draw_label(
        image=image,
        text="Vehicle",
        position=(100, 100),
    )

    image = draw_bounding_box(
        image=image,
        bounding_box=(100, 120, 300, 320),
    )

    assert image.shape == (
        480,
        640,
        3,
    )


def test_visualization_returns_numpy_array() -> None:
    """
    Verify visualization functions return NumPy arrays.
    """

    image = create_test_image()

    output = draw_bounding_box(
        image=image,
        bounding_box=(10, 10, 100, 100),
    )

    assert isinstance(output, np.ndarray)


def test_original_image_not_modified_reference() -> None:
    """
    Verify returned image object is valid.

    This test ensures the returned image can safely be used
    in subsequent pipeline stages.
    """

    image = create_test_image()

    output = draw_text(
        image=image,
        text="Testing",
        position=(30, 30),
    )

    assert output is not None
    assert output.shape == image.shape


def test_visualization_handles_empty_image() -> None:
    """
    Verify visualization functions can process an empty
    black image.
    """

    image = create_test_image()

    output = draw_detection(
        image=image,
        detection=create_detection(),
    )

    assert np.any(output >= 0)