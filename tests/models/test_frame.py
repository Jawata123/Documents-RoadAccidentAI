"""
Unit tests for road_accident_detection.models.frame.

These tests verify the Frame domain model, ensuring correct initialization,
property behavior, serialization, and object independence.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import numpy as np

from road_accident_detection.models.frame import Frame


def create_test_image() -> np.ndarray:
    """
    Create a dummy RGB image for testing.

    Returns:
        NumPy image array.
    """

    return np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )


def test_frame_initialization() -> None:
    """
    Verify Frame initialization.
    """

    image = create_test_image()

    frame = Frame(
        index=1,
        image=image,
        fps=30.0,
        source="video.mp4",
    )

    assert frame.index == 1
    assert frame.image is image
    assert frame.fps == 30.0
    assert frame.source == "video.mp4"


def test_frame_dimensions() -> None:
    """
    Verify frame dimensions.
    """

    frame = Frame(
        index=0,
        image=create_test_image(),
        fps=25.0,
        source="test.mp4",
    )

    assert frame.width == 640
    assert frame.height == 480


def test_frame_channels() -> None:
    """
    Verify image channel count.
    """

    frame = Frame(
        index=0,
        image=create_test_image(),
        fps=30.0,
        source="video.mp4",
    )

    assert frame.channels == 3


def test_frame_shape() -> None:
    """
    Verify image shape.
    """

    frame = Frame(
        index=0,
        image=create_test_image(),
        fps=30.0,
        source="video.mp4",
    )

    assert frame.shape == (480, 640, 3)


def test_frame_size() -> None:
    """
    Verify image size.
    """

    frame = Frame(
        index=0,
        image=create_test_image(),
        fps=30.0,
        source="video.mp4",
    )

    assert frame.size == (640, 480)


def test_frame_copy() -> None:
    """
    Verify frame copy creates a deep copy.
    """

    frame = Frame(
        index=1,
        image=create_test_image(),
        fps=30.0,
        source="video.mp4",
    )

    copied = frame.copy()

    assert copied is not frame
    assert copied.image is not frame.image

    np.testing.assert_array_equal(
        copied.image,
        frame.image,
    )


def test_frame_to_dict() -> None:
    """
    Verify dictionary serialization.
    """

    frame = Frame(
        index=3,
        image=create_test_image(),
        fps=60.0,
        source="sample.mp4",
    )

    data = frame.to_dict()

    assert data["index"] == 3
    assert data["fps"] == 60.0
    assert data["source"] == "sample.mp4"
    assert data["width"] == 640
    assert data["height"] == 480


def test_frame_repr() -> None:
    """
    Verify string representation.
    """

    frame = Frame(
        index=5,
        image=create_test_image(),
        fps=30.0,
        source="traffic.mp4",
    )

    representation = repr(frame)

    assert isinstance(
        representation,
        str,
    )

    assert "Frame" in representation


def test_multiple_frames_are_independent() -> None:
    """
    Verify Frame instances are independent.
    """

    frame1 = Frame(
        index=1,
        image=create_test_image(),
        fps=30.0,
        source="a.mp4",
    )

    frame2 = Frame(
        index=2,
        image=create_test_image(),
        fps=30.0,
        source="b.mp4",
    )

    assert frame1 is not frame2
    assert frame1.image is not frame2.image


def test_frame_image_dtype() -> None:
    """
    Verify image data type.
    """

    frame = Frame(
        index=0,
        image=create_test_image(),
        fps=30.0,
        source="video.mp4",
    )

    assert frame.image.dtype == np.uint8


def test_frame_pixel_count() -> None:
    """
    Verify total pixel count.
    """

    frame = Frame(
        index=0,
        image=create_test_image(),
        fps=30.0,
        source="video.mp4",
    )

    assert (
        frame.width * frame.height
        == 640 * 480
    )