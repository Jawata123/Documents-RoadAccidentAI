"""
Unit tests for road_accident_detection.core.constants.

These tests verify that the project's global constants remain consistent,
correctly typed, and suitable for use throughout the application.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

from road_accident_detection.core.constants import (
    CONFIG_DIRECTORY_NAME,
    DATASET_DIRECTORY_NAME,
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOGGER_NAME,
    DEFAULT_WINDOW_NAME,
    EMPTY_PATH,
    ENV_CONFIG_FILE,
    ENV_LOG_LEVEL,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    LOGGING_CONFIG_FILENAME,
    LOG_DIRECTORY_NAME,
    MILLISECONDS_PER_SECOND,
    MINIMUM_PYTHON_VERSION,
    MODEL_DIRECTORY_NAME,
    OUTPUT_DIRECTORY_NAME,
    PACKAGE_NAME,
    PROJECT_NAME,
    PROJECT_VERSION,
    SECONDS_PER_MINUTE,
    SRC_DIRECTORY_NAME,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    TEST_DIRECTORY_NAME,
    VEHICLE_CLASS_NAMES,
)


def test_project_information() -> None:
    """Verify project metadata constants."""

    assert PROJECT_NAME == "RoadAccidentAI"
    assert PACKAGE_NAME == "road_accident_detection"
    assert isinstance(PROJECT_VERSION, str)
    assert PROJECT_VERSION


def test_python_version() -> None:
    """Verify minimum supported Python version."""

    assert isinstance(MINIMUM_PYTHON_VERSION, tuple)
    assert len(MINIMUM_PYTHON_VERSION) == 2

    major, minor = MINIMUM_PYTHON_VERSION

    assert isinstance(major, int)
    assert isinstance(minor, int)
    assert major >= 3


def test_directory_names() -> None:
    """Verify directory name constants."""

    assert SRC_DIRECTORY_NAME == "src"
    assert CONFIG_DIRECTORY_NAME == "configs"
    assert MODEL_DIRECTORY_NAME == "models"
    assert DATASET_DIRECTORY_NAME == "datasets"
    assert LOG_DIRECTORY_NAME == "logs"
    assert OUTPUT_DIRECTORY_NAME == "outputs"
    assert TEST_DIRECTORY_NAME == "tests"


def test_configuration_files() -> None:
    """Verify configuration filenames."""

    assert DEFAULT_CONFIG_FILENAME.endswith(".yaml")
    assert LOGGING_CONFIG_FILENAME.endswith(".yaml")


def test_logging_constants() -> None:
    """Verify logging-related constants."""

    assert DEFAULT_LOG_LEVEL == "INFO"
    assert DEFAULT_LOGGER_NAME == PROJECT_NAME
    assert DEFAULT_LOG_FILE.endswith(".log")
    assert "%(levelname)" in DEFAULT_LOG_FORMAT
    assert DEFAULT_WINDOW_NAME == PROJECT_NAME


def test_supported_image_extensions() -> None:
    """Verify supported image formats."""

    assert isinstance(
        SUPPORTED_IMAGE_EXTENSIONS,
        frozenset,
    )

    assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".bmp" in SUPPORTED_IMAGE_EXTENSIONS


def test_supported_video_extensions() -> None:
    """Verify supported video formats."""

    assert isinstance(
        SUPPORTED_VIDEO_EXTENSIONS,
        frozenset,
    )

    assert ".mp4" in SUPPORTED_VIDEO_EXTENSIONS
    assert ".avi" in SUPPORTED_VIDEO_EXTENSIONS
    assert ".mov" in SUPPORTED_VIDEO_EXTENSIONS
    assert ".mkv" in SUPPORTED_VIDEO_EXTENSIONS


def test_vehicle_class_names() -> None:
    """Verify supported vehicle classes."""

    expected = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "motorbike",
        "bicycle",
    }

    assert expected.issubset(VEHICLE_CLASS_NAMES)


def test_environment_variables() -> None:
    """Verify environment variable names."""

    assert ENV_CONFIG_FILE == "ROAD_ACCIDENT_CONFIG"
    assert ENV_LOG_LEVEL == "ROAD_ACCIDENT_LOG_LEVEL"


def test_time_constants() -> None:
    """Verify time conversion constants."""

    assert MILLISECONDS_PER_SECOND == 1000
    assert SECONDS_PER_MINUTE == 60


def test_exit_codes() -> None:
    """Verify process exit codes."""

    assert EXIT_SUCCESS == 0
    assert EXIT_FAILURE == 1


def test_empty_path() -> None:
    """Verify empty path constant."""

    assert isinstance(EMPTY_PATH, Path)
    assert EMPTY_PATH == Path()