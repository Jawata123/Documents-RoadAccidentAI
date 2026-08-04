"""
Unit tests for road_accident_detection.core.exceptions.

These tests verify the project's custom exception hierarchy and ensure that
all exceptions inherit from the correct base classes.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import pytest

from road_accident_detection.core.exceptions import (
    ConfigurationError,
    DetectionError,
    FileNotFoundError,
    FrameReadError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    PathError,
    PipelineError,
    RoadAccidentAIError,
    ValidationError,
    VideoError,
    VideoOpenError,
)


def test_base_exception_inheritance() -> None:
    """Verify the base project exception."""

    assert issubclass(
        RoadAccidentAIError,
        Exception,
    )


def test_configuration_error_inheritance() -> None:
    """Verify ConfigurationError inheritance."""

    assert issubclass(
        ConfigurationError,
        RoadAccidentAIError,
    )


def test_validation_error_inheritance() -> None:
    """Verify ValidationError inheritance."""

    assert issubclass(
        ValidationError,
        RoadAccidentAIError,
    )


def test_path_error_inheritance() -> None:
    """Verify PathError inheritance."""

    assert issubclass(
        PathError,
        RoadAccidentAIError,
    )


def test_file_not_found_error_inheritance() -> None:
    """Verify custom FileNotFoundError inheritance."""

    assert issubclass(
        FileNotFoundError,
        PathError,
    )

    assert issubclass(
        FileNotFoundError,
        RoadAccidentAIError,
    )


def test_model_error_hierarchy() -> None:
    """Verify model exception hierarchy."""

    assert issubclass(
        ModelError,
        RoadAccidentAIError,
    )

    assert issubclass(
        ModelLoadError,
        ModelError,
    )

    assert issubclass(
        ModelInferenceError,
        ModelError,
    )


def test_video_error_hierarchy() -> None:
    """Verify video exception hierarchy."""

    assert issubclass(
        VideoError,
        RoadAccidentAIError,
    )

    assert issubclass(
        VideoOpenError,
        VideoError,
    )

    assert issubclass(
        FrameReadError,
        VideoError,
    )


def test_detection_error_inheritance() -> None:
    """Verify DetectionError inheritance."""

    assert issubclass(
        DetectionError,
        RoadAccidentAIError,
    )


def test_pipeline_error_inheritance() -> None:
    """Verify PipelineError inheritance."""

    assert issubclass(
        PipelineError,
        RoadAccidentAIError,
    )


@pytest.mark.parametrize(
    "exception_type",
    [
        RoadAccidentAIError,
        ConfigurationError,
        ValidationError,
        PathError,
        FileNotFoundError,
        ModelError,
        ModelLoadError,
        ModelInferenceError,
        VideoError,
        VideoOpenError,
        FrameReadError,
        DetectionError,
        PipelineError,
    ],
)
def test_exception_message(
    exception_type: type[Exception],
) -> None:
    """
    Verify exception message preservation.

    Args:
        exception_type:
            Exception class under test.
    """

    message = "Test exception message"

    exception = exception_type(message)

    assert str(exception) == message


@pytest.mark.parametrize(
    "exception_type",
    [
        ConfigurationError,
        ValidationError,
        PathError,
        FileNotFoundError,
        ModelError,
        ModelLoadError,
        ModelInferenceError,
        VideoError,
        VideoOpenError,
        FrameReadError,
        DetectionError,
        PipelineError,
    ],
)
def test_exception_is_project_exception(
    exception_type: type[RoadAccidentAIError],
) -> None:
    """
    Verify every custom exception is catchable via RoadAccidentAIError.

    Args:
        exception_type:
            Exception class under test.
    """

    with pytest.raises(RoadAccidentAIError):
        raise exception_type("failure")