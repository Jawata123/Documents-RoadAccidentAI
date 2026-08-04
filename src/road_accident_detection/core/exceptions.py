"""
Custom exception hierarchy for RoadAccidentAI.

This module defines all project-specific exceptions. Every package in the
project should raise these exceptions instead of generic Python exceptions
whenever possible.

The exception hierarchy is intentionally designed for future extensibility.
As additional research modules (tracking, feature extraction, evaluation,
dashboard, REST API, etc.) are introduced, they can derive from the base
project exception without breaking compatibility.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

__all__ = [
    "RoadAccidentAIError",
    "ConfigurationError",
    "ValidationError",
    "PathError",
    "FileNotFoundError",
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "VideoError",
    "VideoOpenError",
    "FrameReadError",
    "DetectionError",
    "PipelineError",
]


class RoadAccidentAIError(Exception):
    """
    Base exception for the entire RoadAccidentAI project.

    Every custom exception in the project should inherit from this class.
    This allows client code to catch all project-specific exceptions using
    a single exception type when appropriate.
    """


###############################################################################
# Configuration Exceptions
###############################################################################


class ConfigurationError(RoadAccidentAIError):
    """
    Raised when configuration loading or parsing fails.
    """


class ValidationError(RoadAccidentAIError):
    """
    Raised when configuration or input validation fails.
    """


###############################################################################
# Filesystem Exceptions
###############################################################################


class PathError(RoadAccidentAIError):
    """
    Raised when a filesystem path is invalid or inaccessible.
    """


class FileNotFoundError(PathError):
    """
    Raised when a required project file cannot be found.

    Note:
        This intentionally subclasses PathError instead of Python's built-in
        FileNotFoundError to keep all project exceptions within the project's
        exception hierarchy.
    """


###############################################################################
# Model Exceptions
###############################################################################


class ModelError(RoadAccidentAIError):
    """
    Base exception for model-related failures.
    """


class ModelLoadError(ModelError):
    """
    Raised when the detection model cannot be loaded.
    """


class ModelInferenceError(ModelError):
    """
    Raised when inference fails during model execution.
    """


###############################################################################
# Video Exceptions
###############################################################################


class VideoError(RoadAccidentAIError):
    """
    Base exception for video processing failures.
    """


class VideoOpenError(VideoError):
    """
    Raised when a video source cannot be opened.
    """


class FrameReadError(VideoError):
    """
    Raised when a frame cannot be read from the video source.
    """


###############################################################################
# Detection Exceptions
###############################################################################


class DetectionError(RoadAccidentAIError):
    """
    Raised when an error occurs during object detection.
    """


###############################################################################
# Pipeline Exceptions
###############################################################################


class PipelineError(RoadAccidentAIError):
    """
    Raised when the processing pipeline encounters an unrecoverable error.
    """