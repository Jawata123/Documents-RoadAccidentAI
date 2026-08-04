"""
Unit tests for road_accident_detection.config.schemas.

These tests verify the project's configuration schema models. The schema
layer defines the structure and validation rules for application
configuration, ensuring that configuration objects are correctly
constructed and validated.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import pytest

from road_accident_detection.config.schemas import (
    ApplicationConfig,
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    VideoConfig,
)


def test_application_config_defaults() -> None:
    """
    Verify ApplicationConfig default values.
    """

    config = ApplicationConfig()

    assert config.name
    assert config.version


def test_logging_config_defaults() -> None:
    """
    Verify LoggingConfig default values.
    """

    config = LoggingConfig()

    assert config.level in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    assert isinstance(config.save_to_file, bool)
    assert config.file_name.endswith(".log")


def test_model_config_defaults() -> None:
    """
    Verify ModelConfig default values.
    """

    config = ModelConfig()

    assert config.detector
    assert config.weights
    assert config.device in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }

    assert config.image_size > 0
    assert 0.0 <= config.confidence <= 1.0


def test_video_config_defaults() -> None:
    """
    Verify VideoConfig default values.
    """

    config = VideoConfig()

    assert config.source
    assert isinstance(config.display, bool)
    assert isinstance(config.loop, bool)


def test_output_config_defaults() -> None:
    """
    Verify OutputConfig default values.
    """

    config = OutputConfig()

    assert config.directory
    assert isinstance(config.save_video, bool)
    assert isinstance(config.save_frames, bool)


def test_application_config_custom_values() -> None:
    """
    Verify ApplicationConfig accepts custom values.
    """

    config = ApplicationConfig(
        name="TestProject",
        version="1.2.3",
    )

    assert config.name == "TestProject"
    assert config.version == "1.2.3"


def test_logging_config_custom_values() -> None:
    """
    Verify LoggingConfig accepts custom values.
    """

    config = LoggingConfig(
        level="DEBUG",
        save_to_file=False,
        file_name="debug.log",
    )

    assert config.level == "DEBUG"
    assert config.save_to_file is False
    assert config.file_name == "debug.log"


def test_model_config_custom_values() -> None:
    """
    Verify ModelConfig accepts custom values.
    """

    config = ModelConfig(
        detector="ultralytics",
        weights="models/custom.pt",
        device="cpu",
        image_size=1280,
        confidence=0.55,
    )

    assert config.detector == "ultralytics"
    assert config.weights == "models/custom.pt"
    assert config.device == "cpu"
    assert config.image_size == 1280
    assert config.confidence == 0.55


def test_video_config_custom_values() -> None:
    """
    Verify VideoConfig accepts custom values.
    """

    config = VideoConfig(
        source="datasets/demo.mp4",
        display=False,
        loop=True,
    )

    assert config.source == "datasets/demo.mp4"
    assert config.display is False
    assert config.loop is True


def test_output_config_custom_values() -> None:
    """
    Verify OutputConfig accepts custom values.
    """

    config = OutputConfig(
        directory="results",
        save_video=True,
        save_frames=True,
    )

    assert config.directory == "results"
    assert config.save_video is True
    assert config.save_frames is True


@pytest.mark.parametrize(
    "confidence",
    [
        0.0,
        0.25,
        0.5,
        1.0,
    ],
)
def test_model_config_confidence_range(
    confidence: float,
) -> None:
    """
    Verify valid confidence values.

    Args:
        confidence:
            Confidence value under test.
    """

    config = ModelConfig(
        confidence=confidence,
    )

    assert config.confidence == confidence


def test_schema_instances_are_independent() -> None:
    """
    Verify separate schema instances do not share state.
    """

    first = ModelConfig()
    second = ModelConfig()

    assert first is not second