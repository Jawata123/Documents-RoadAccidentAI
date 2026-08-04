"""
Unit tests for road_accident_detection.config.defaults.

These tests verify the project's default configuration values. The defaults
module provides the baseline configuration used when user-defined settings
are not supplied.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from road_accident_detection.config.defaults import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_SOURCE,
    DEFAULT_VIDEO_EXTENSIONS,
    DEFAULT_WEBCAM_INDEX,
)


def test_default_model_name() -> None:
    """
    Verify the default model name.
    """

    assert isinstance(DEFAULT_MODEL_NAME, str)
    assert DEFAULT_MODEL_NAME
    assert DEFAULT_MODEL_NAME.endswith(".pt")


def test_default_model_path() -> None:
    """
    Verify the default model path.
    """

    assert str(DEFAULT_MODEL_PATH).endswith(
        str(DEFAULT_MODEL_NAME)
    )


def test_default_device() -> None:
    """
    Verify the default inference device.
    """

    assert DEFAULT_DEVICE in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }


def test_default_image_size() -> None:
    """
    Verify the default image size.
    """

    assert isinstance(DEFAULT_IMAGE_SIZE, int)
    assert DEFAULT_IMAGE_SIZE > 0
    assert DEFAULT_IMAGE_SIZE % 32 == 0


def test_default_confidence_threshold() -> None:
    """
    Verify the default confidence threshold.
    """

    assert isinstance(
        DEFAULT_CONFIDENCE_THRESHOLD,
        float,
    )

    assert 0.0 <= DEFAULT_CONFIDENCE_THRESHOLD <= 1.0


def test_default_log_level() -> None:
    """
    Verify the default logging level.
    """

    assert DEFAULT_LOG_LEVEL in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }


def test_default_output_directory() -> None:
    """
    Verify the default output directory.
    """

    assert DEFAULT_OUTPUT_DIRECTORY.name == "outputs"


def test_default_source() -> None:
    """
    Verify the default video source.
    """

    assert isinstance(DEFAULT_SOURCE, str)


def test_default_webcam_index() -> None:
    """
    Verify the default webcam index.
    """

    assert isinstance(DEFAULT_WEBCAM_INDEX, int)
    assert DEFAULT_WEBCAM_INDEX >= 0


def test_default_video_extensions() -> None:
    """
    Verify supported video extensions.
    """

    assert isinstance(
        DEFAULT_VIDEO_EXTENSIONS,
        frozenset,
    )

    expected = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }

    assert expected.issubset(
        DEFAULT_VIDEO_EXTENSIONS
    )


def test_default_values_are_not_empty() -> None:
    """
    Verify essential default values are populated.
    """

    assert DEFAULT_MODEL_NAME
    assert DEFAULT_DEVICE
    assert DEFAULT_LOG_LEVEL
    assert DEFAULT_OUTPUT_DIRECTORY