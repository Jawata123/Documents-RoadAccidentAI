"""
Configuration schema definitions for RoadAccidentAI.

This module defines strongly typed configuration models used throughout the
application. All configuration loaded from YAML files is validated against
these schemas before becoming available to the rest of the system.

The schemas intentionally represent only the base framework. Research-specific
configuration (tracking, trajectory analysis, accident logic, etc.) will be
added later without breaking the existing API.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


__all__ = [
    "LoggingConfig",
    "ModelConfig",
    "VideoConfig",
    "OutputConfig",
    "ApplicationConfig",
]


###############################################################################
# Logging
###############################################################################


class LoggingConfig(BaseModel):
    """
    Logging configuration.

    Attributes:
        level:
            Logging level.

        save_to_file:
            Enable file logging.

        file_name:
            Log filename.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    level: str = Field(
        default="INFO",
        description="Logging level.",
    )

    save_to_file: bool = Field(
        default=True,
        description="Write logs to a file.",
    )

    file_name: str = Field(
        default="road_accident_ai.log",
        description="Log filename.",
    )


###############################################################################
# Model
###############################################################################


class ModelConfig(BaseModel):
    """
    YOLO model configuration.

    Attributes:
        weights:
            Path to the model weights.

        device:
            Inference device.

        image_size:
            Input image size.

        confidence:
            Detection confidence threshold.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    weights: Path = Field(
        default=Path("models/yolo.pt"),
    )

    device: str = Field(
        default="auto",
    )

    image_size: PositiveInt = Field(
        default=640,
        ge=320,
        le=2048,
    )

    confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )


###############################################################################
# Video
###############################################################################


class VideoConfig(BaseModel):
    """
    Video processing configuration.

    Attributes:
        source:
            Video source.

        loop:
            Restart video when finished.

        display:
            Display processed frames.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    source: str = Field(
        default="0",
    )

    loop: bool = Field(
        default=False,
    )

    display: bool = Field(
        default=True,
    )


###############################################################################
# Output
###############################################################################


class OutputConfig(BaseModel):
    """
    Output configuration.

    Attributes:
        directory:
            Output directory.

        save_video:
            Save processed video.

        save_frames:
            Save processed frames.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    directory: Path = Field(
        default=Path("outputs"),
    )

    save_video: bool = Field(
        default=False,
    )

    save_frames: bool = Field(
        default=False,
    )


###############################################################################
# Root Configuration
###############################################################################


class ApplicationConfig(BaseModel):
    """
    Root application configuration.

    This is the top-level configuration object used throughout the project.
    Every configuration file is validated into this model before use.

    Attributes:
        logging:
            Logging configuration.

        model:
            YOLO model configuration.

        video:
            Video configuration.

        output:
            Output configuration.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
    )

    model: ModelConfig = Field(
        default_factory=ModelConfig,
    )

    video: VideoConfig = Field(
        default_factory=VideoConfig,
    )

    output: OutputConfig = Field(
        default_factory=OutputConfig,
    )