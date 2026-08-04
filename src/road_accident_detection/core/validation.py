"""
Validation utilities for RoadAccidentAI.

This module provides reusable validation helpers that are shared across the
entire project. The functions here perform common validation tasks without
depending on any computer vision, deep learning, or application-specific
logic.

The goal is to centralize validation so every package follows the same rules.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    MINIMUM_PYTHON_VERSION,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from .exceptions import (
    ConfigurationError,
    FileNotFoundError,
    PathError,
    ValidationError,
)
from .types import PathLike

__all__ = [
    "validate_python_version",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_image_file",
    "validate_video_file",
    "validate_not_none",
    "validate_not_empty",
]


###############################################################################
# Python Validation
###############################################################################


def validate_python_version() -> None:
    """
    Validate the running Python version.

    Raises:
        ConfigurationError:
            If the running Python version is lower than the minimum supported
            version.
    """

    if sys.version_info < MINIMUM_PYTHON_VERSION:
        required = ".".join(map(str, MINIMUM_PYTHON_VERSION))
        current = ".".join(map(str, sys.version_info[:3]))

        raise ConfigurationError(
            f"Python {required}+ is required. "
            f"Current version: {current}"
        )


###############################################################################
# Path Validation
###############################################################################


def validate_file_exists(path: PathLike) -> Path:
    """
    Validate that a file exists.

    Args:
        path:
            File path.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        PathError:
            If the path is not a file.
    """

    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise PathError(
            f"Expected a file but received: {file_path}"
        )

    return file_path


def validate_directory_exists(path: PathLike) -> Path:
    """
    Validate that a directory exists.

    Args:
        path:
            Directory path.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError:
            If the directory does not exist.

        PathError:
            If the path is not a directory.
    """

    directory = Path(path).expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise PathError(
            f"Expected a directory but received: {directory}"
        )

    return directory


###############################################################################
# Media Validation
###############################################################################


def validate_image_file(path: PathLike) -> Path:
    """
    Validate an image file.

    Args:
        path:
            Image file path.

    Returns:
        Validated image path.

    Raises:
        ValidationError:
            If the file extension is unsupported.
    """

    image = validate_file_exists(path)

    if image.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValidationError(
            f"Unsupported image format: {image.suffix}"
        )

    return image


def validate_video_file(path: PathLike) -> Path:
    """
    Validate a video file.

    Args:
        path:
            Video file path.

    Returns:
        Validated video path.

    Raises:
        ValidationError:
            If the file extension is unsupported.
    """

    video = validate_file_exists(path)

    if video.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValidationError(
            f"Unsupported video format: {video.suffix}"
        )

    return video


###############################################################################
# Generic Validation
###############################################################################


def validate_not_none(
    value: Any,
    name: str,
) -> None:
    """
    Validate that a value is not None.

    Args:
        value:
            Value to validate.

        name:
            Variable name.

    Raises:
        ValidationError:
            If the value is None.
    """

    if value is None:
        raise ValidationError(
            f"'{name}' cannot be None."
        )


def validate_not_empty(
    value: str | Iterable[Any],
    name: str,
) -> None:
    """
    Validate that an iterable or string is not empty.

    Args:
        value:
            Object to validate.

        name:
            Variable name.

    Raises:
        ValidationError:
            If the object is empty.
    """

    if len(value) == 0:
        raise ValidationError(
            f"'{name}' cannot be empty."
        )