"""
Image utility functions for RoadAccidentAI.

This module provides reusable image-processing utilities used throughout the
project. These functions are intentionally generic and independent of YOLO,
tracking, or accident detection logic.

The utilities operate on NumPy arrays (OpenCV images) and provide common
operations required by multiple modules.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from road_accident_detection.core.types import Size2D

__all__ = [
    "resize_image",
    "resize_keep_aspect_ratio",
    "convert_bgr_to_rgb",
    "convert_rgb_to_bgr",
    "convert_to_grayscale",
    "normalize_image",
    "copy_image",
]

_IMAGE_MAX_VALUE: Final[float] = 255.0


def resize_image(
    image: NDArray[np.uint8],
    size: Size2D,
    interpolation: int = cv2.INTER_LINEAR,
) -> NDArray[np.uint8]:
    """
    Resize an image.

    Args:
        image:
            Input image.

        size:
            Target size as (width, height).

        interpolation:
            OpenCV interpolation method.

    Returns:
        Resized image.
    """

    return cv2.resize(
        image,
        size,
        interpolation=interpolation,
    )


def resize_keep_aspect_ratio(
    image: NDArray[np.uint8],
    max_size: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> NDArray[np.uint8]:
    """
    Resize an image while preserving its aspect ratio.

    The image is scaled so that its longest side equals ``max_size``.

    Args:
        image:
            Input image.

        max_size:
            Maximum width or height.

        interpolation:
            OpenCV interpolation method.

    Returns:
        Resized image.
    """

    height, width = image.shape[:2]

    longest_side = max(width, height)

    if longest_side <= max_size:
        return image.copy()

    scale = max_size / float(longest_side)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )


def convert_bgr_to_rgb(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """
    Convert a BGR image to RGB.

    Args:
        image:
            BGR image.

    Returns:
        RGB image.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def convert_rgb_to_bgr(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """
    Convert an RGB image to BGR.

    Args:
        image:
            RGB image.

    Returns:
        BGR image.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )


def convert_to_grayscale(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """
    Convert an image to grayscale.

    Args:
        image:
            Input image.

    Returns:
        Single-channel grayscale image.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def normalize_image(
    image: NDArray[np.uint8],
) -> NDArray[np.float32]:
    """
    Normalize an image to the range [0.0, 1.0].

    Args:
        image:
            Input image.

    Returns:
        Normalized float32 image.
    """

    return image.astype(np.float32) / _IMAGE_MAX_VALUE


def copy_image(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """
    Create a deep copy of an image.

    Args:
        image:
            Input image.

    Returns:
        Copied image.
    """

    return image.copy()