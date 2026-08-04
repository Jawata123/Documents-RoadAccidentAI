"""
Unit tests for road_accident_detection.core.validation.

These tests verify the project's validation utilities. The validation layer
is responsible for checking inputs before they are consumed by the pipeline,
ensuring invalid data is rejected early with clear exceptions.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from road_accident_detection.core.exceptions import ValidationError
from road_accident_detection.core.validation import (
    validate_confidence,
    validate_directory,
    validate_file,
    validate_frame,
    validate_image_size,
    validate_model_path,
)


def test_validate_confidence_accepts_valid_values() -> None:
    """
    Verify valid confidence values are accepted.
    """

    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(0.25) == 0.25
    assert validate_confidence(0.5) == 0.5
    assert validate_confidence(1.0) == 1.0


@pytest.mark.parametrize(
    "confidence",
    [
        -0.1,
        -1.0,
        1.1,
        2.0,
    ],
)
def test_validate_confidence_rejects_invalid_values(
    confidence: float,
) -> None:
    """
    Verify invalid confidence values raise ValidationError.

    Args:
        confidence:
            Confidence value under test.
    """

    with pytest.raises(ValidationError):
        validate_confidence(confidence)


def test_validate_image_size_accepts_valid_sizes() -> None:
    """
    Verify valid image sizes are accepted.
    """

    assert validate_image_size(320) == 320
    assert validate_image_size(640) == 640
    assert validate_image_size(1280) == 1280


@pytest.mark.parametrize(
    "size",
    [
        0,
        -1,
        -640,
    ],
)
def test_validate_image_size_rejects_invalid_sizes(
    size: int,
) -> None:
    """
    Verify invalid image sizes raise ValidationError.

    Args:
        size:
            Image size under test.
    """

    with pytest.raises(ValidationError):
        validate_image_size(size)


def test_validate_directory_accepts_existing_directory(
    tmp_path: Path,
) -> None:
    """
    Verify an existing directory is accepted.

    Args:
        tmp_path:
            Temporary directory supplied by pytest.
    """

    result = validate_directory(tmp_path)

    assert result == tmp_path


def test_validate_directory_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """
    Verify a missing directory raises ValidationError.

    Args:
        tmp_path:
            Temporary directory.
    """

    directory = tmp_path / "missing"

    with pytest.raises(ValidationError):
        validate_directory(directory)


def test_validate_file_accepts_existing_file(
    tmp_path: Path,
) -> None:
    """
    Verify an existing file is accepted.

    Args:
        tmp_path:
            Temporary directory.
    """

    file_path = tmp_path / "sample.txt"
    file_path.write_text("RoadAccidentAI")

    result = validate_file(file_path)

    assert result == file_path


def test_validate_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    Verify a missing file raises ValidationError.

    Args:
        tmp_path:
            Temporary directory.
    """

    file_path = tmp_path / "missing.txt"

    with pytest.raises(ValidationError):
        validate_file(file_path)


def test_validate_model_path_accepts_existing_model(
    tmp_path: Path,
) -> None:
    """
    Verify an existing model file is accepted.

    Args:
        tmp_path:
            Temporary directory.
    """

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")

    result = validate_model_path(model_path)

    assert result == model_path


def test_validate_model_path_rejects_missing_model(
    tmp_path: Path,
) -> None:
    """
    Verify a missing model file raises ValidationError.

    Args:
        tmp_path:
            Temporary directory.
    """

    model_path = tmp_path / "missing.pt"

    with pytest.raises(ValidationError):
        validate_model_path(model_path)


def test_validate_frame_accepts_valid_image() -> None:
    """
    Verify a valid image frame is accepted.
    """

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    result = validate_frame(image)

    assert result is image


def test_validate_frame_rejects_none() -> None:
    """
    Verify None is rejected as a frame.
    """

    with pytest.raises(ValidationError):
        validate_frame(None)


def test_validate_frame_rejects_empty_image() -> None:
    """
    Verify an empty image is rejected.
    """

    image = np.empty(
        (0, 0, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ValidationError):
        validate_frame(image)


def test_validate_frame_rejects_invalid_dimension() -> None:
    """
    Verify an invalid frame shape is rejected.
    """

    image = np.array(
        [],
        dtype=np.uint8,
    )

    with pytest.raises(ValidationError):
        validate_frame(image)