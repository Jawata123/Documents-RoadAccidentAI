"""
Global constants used throughout the RoadAccidentAI project.

This module centralizes application-wide constants to avoid magic numbers,
hard-coded strings, and duplicated values across the codebase.

The constants defined here are intended to remain stable and independent of
specific research algorithms. Accident detection thresholds, tracking
parameters, and experiment-specific values should NOT be placed here; those
belong in configuration files.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

###############################################################################
# Project Information
###############################################################################

PROJECT_NAME: Final[str] = "RoadAccidentAI"

PACKAGE_NAME: Final[str] = "road_accident_detection"

PROJECT_VERSION: Final[str] = "0.1.0"

###############################################################################
# Supported Python Version
###############################################################################

MINIMUM_PYTHON_VERSION: Final[tuple[int, int]] = (3, 13)

###############################################################################
# Supported Image Extensions
###############################################################################

SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".bmp",
        ".dib",
        ".jpeg",
        ".jpg",
        ".jpe",
        ".jp2",
        ".png",
        ".webp",
        ".pbm",
        ".pgm",
        ".ppm",
        ".sr",
        ".ras",
        ".tiff",
        ".tif",
    }
)

###############################################################################
# Supported Video Extensions
###############################################################################

SUPPORTED_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".mpeg",
        ".mpg",
        ".m4v",
        ".webm",
        ".ts",
        ".flv",
    }
)

###############################################################################
# Supported YOLO Vehicle Classes
###############################################################################

VEHICLE_CLASS_NAMES: Final[frozenset[str]] = frozenset(
    {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "motorbike",
        "bicycle",
    }
)

###############################################################################
# Default Directory Names
###############################################################################

SRC_DIRECTORY_NAME: Final[str] = "src"

CONFIG_DIRECTORY_NAME: Final[str] = "configs"

MODEL_DIRECTORY_NAME: Final[str] = "models"

DATASET_DIRECTORY_NAME: Final[str] = "datasets"

LOG_DIRECTORY_NAME: Final[str] = "logs"

OUTPUT_DIRECTORY_NAME: Final[str] = "outputs"

TEST_DIRECTORY_NAME: Final[str] = "tests"

###############################################################################
# Default Configuration Files
###############################################################################

DEFAULT_CONFIG_FILENAME: Final[str] = "default.yaml"

LOGGING_CONFIG_FILENAME: Final[str] = "logging.yaml"

###############################################################################
# Logging
###############################################################################

DEFAULT_LOGGER_NAME: Final[str] = PROJECT_NAME

DEFAULT_LOG_LEVEL: Final[str] = "INFO"

DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

###############################################################################
# OpenCV Window
###############################################################################

DEFAULT_WINDOW_NAME: Final[str] = "RoadAccidentAI"

###############################################################################
# File Names
###############################################################################

DEFAULT_LOG_FILE: Final[str] = "road_accident_ai.log"

###############################################################################
# Environment Variables
###############################################################################

ENV_CONFIG_FILE: Final[str] = "ROAD_ACCIDENT_CONFIG"

ENV_LOG_LEVEL: Final[str] = "ROAD_ACCIDENT_LOG_LEVEL"

###############################################################################
# Time
###############################################################################

MILLISECONDS_PER_SECOND: Final[int] = 1000

SECONDS_PER_MINUTE: Final[int] = 60

###############################################################################
# Numerical Defaults
###############################################################################

EPSILON: Final[float] = 1e-8

###############################################################################
# Common Exit Codes
###############################################################################

EXIT_SUCCESS: Final[int] = 0

EXIT_FAILURE: Final[int] = 1

###############################################################################
# Empty Path Constant
###############################################################################

EMPTY_PATH: Final[Path] = Path()

###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "PROJECT_NAME",
    "PACKAGE_NAME",
    "PROJECT_VERSION",
    "MINIMUM_PYTHON_VERSION",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "VEHICLE_CLASS_NAMES",
    "SRC_DIRECTORY_NAME",
    "CONFIG_DIRECTORY_NAME",
    "MODEL_DIRECTORY_NAME",
    "DATASET_DIRECTORY_NAME",
    "LOG_DIRECTORY_NAME",
    "OUTPUT_DIRECTORY_NAME",
    "TEST_DIRECTORY_NAME",
    "DEFAULT_CONFIG_FILENAME",
    "LOGGING_CONFIG_FILENAME",
    "DEFAULT_LOGGER_NAME",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_WINDOW_NAME",
    "DEFAULT_LOG_FILE",
    "ENV_CONFIG_FILE",
    "ENV_LOG_LEVEL",
    "MILLISECONDS_PER_SECOND",
    "SECONDS_PER_MINUTE",
    "EPSILON",
    "EXIT_SUCCESS",
    "EXIT_FAILURE",
    "EMPTY_PATH",
]