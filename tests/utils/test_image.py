"""
Unit tests for road_accident_detection.utils.image.

These tests verify the project's image utility functions used for common
image-processing operations throughout the RoadAccidentAI pipeline.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import numpy as np

from road_accident_detection.utils.image import (
    copy_image,
    image_center,
    image_size,
    is_color_image,
    resize_image,
)


def create_test_image() -> np.ndarray:
    """
    Create a reusable RGB test image.

    Returns:
        RGB NumPy image.
    """

    return np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )


def test_image_size() -> None:
    """
    Verify image size.
    """

    image = create_test_image()

    width, height = image_size(image)

    assert width == 640
    assert height == 480


def test_image_center() -> None:
    """
    Verify image center.
    """

    image = create_test_image()

    center = image_center(image)

    assert center == (
        320,
        240,
    )


def test_copy_image() -> None:
    """
    Verify deep image copy.
    """

    image = create_test_image()

    copied = copy_image(image)

    assert copied is not image

    np.testing.assert_array_equal(
        copied,
        image,
    )


def test_copy_image_independence() -> None:
    """
    Verify copied image is independent.
    """

    image = create_test_image()

    copied = copy_image(image)

    copied[0, 0, 0] = 255

    assert image[0, 0, 0] == 0


def test_resize_image() -> None:
    """
    Verify image resizing.
    """

    image = create_test_image()

    resized = resize_image(
        image,
        width=320,
        height=240,
    )

    assert resized.shape == (
        240,
        320,
        3,
    )


def test_resize_preserves_dtype() -> None:
    """
    Verify resizing preserves image data type.
    """

    image = create_test_image()

    resized = resize_image(
        image,
        width=200,
        height=100,
    )

    assert resized.dtype == image.dtype


def test_is_color_image_true() -> None:
    """
    Verify RGB image detection.
    """

    image = create_test_image()

    assert is_color_image(image) is True


def test_is_color_image_false() -> None:
    """
    Verify grayscale image detection.
    """

    image = np.zeros(
        (480, 640),
        dtype=np.uint8,
    )

    assert is_color_image(image) is False


def test_multiple_image_copies_are_independent() -> None:
    """
    Verify multiple copies are independent.
    """

    image = create_test_image()

    copy1 = copy_image(image)
    copy2 = copy_image(image)

    assert copy1 is not copy2

    copy1[0, 0, 0] = 100

    assert copy2[0, 0, 0] == 0


def test_image_dimensions_after_resize() -> None:
    """
    Verify resized image dimensions.
    """

    image = create_test_image()

    resized = resize_image(
        image,
        width=1280,
        height=720,
    )

    width, height = image_size(resized)

    assert width == 1280
    assert height == 720


def test_image_center_after_resize() -> None:
    """
    Verify center after resizing.
    """

    image = create_test_image()

    resized = resize_image(
        image,
        width=800,
        height=600,
    )

    assert image_center(resized) == (
        400,
        300,
    )


def test_image_array_type() -> None:
    """
    Verify returned object is a NumPy array.
    """

    image = create_test_image()

    assert isinstance(image, np.ndarray)