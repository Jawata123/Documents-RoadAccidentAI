"""
Unit tests for road_accident_detection.config.settings.

These tests verify the application's settings object. The settings layer
provides a centralized, validated configuration interface for the rest of
the application.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from road_accident_detection.config.settings import Settings


def test_settings_default_initialization() -> None:
    """
    Verify Settings can be created with default values.
    """

    settings = Settings()

    assert settings.application is not None
    assert settings.logging is not None
    assert settings.model is not None
    assert settings.video is not None
    assert settings.output is not None


def test_application_settings_exist() -> None:
    """
    Verify application configuration exists.
    """

    settings = Settings()

    assert settings.application.name
    assert settings.application.version


def test_logging_settings_exist() -> None:
    """
    Verify logging configuration exists.
    """

    settings = Settings()

    assert settings.logging.level
    assert isinstance(
        settings.logging.save_to_file,
        bool,
    )


def test_model_settings_exist() -> None:
    """
    Verify model configuration exists.
    """

    settings = Settings()

    assert settings.model.detector
    assert settings.model.weights
    assert settings.model.device
    assert settings.model.image_size > 0
    assert 0.0 <= settings.model.confidence <= 1.0


def test_video_settings_exist() -> None:
    """
    Verify video configuration exists.
    """

    settings = Settings()

    assert settings.video.source
    assert isinstance(settings.video.display, bool)
    assert isinstance(settings.video.loop, bool)


def test_output_settings_exist() -> None:
    """
    Verify output configuration exists.
    """

    settings = Settings()

    assert settings.output.directory
    assert isinstance(
        settings.output.save_video,
        bool,
    )
    assert isinstance(
        settings.output.save_frames,
        bool,
    )


def test_settings_instances_are_independent() -> None:
    """
    Verify Settings instances do not share mutable state.
    """

    first = Settings()
    second = Settings()

    assert first is not second

    assert first.application is not second.application
    assert first.logging is not second.logging
    assert first.model is not second.model
    assert first.video is not second.video
    assert first.output is not second.output


def test_model_configuration_constraints() -> None:
    """
    Verify model configuration constraints.
    """

    settings = Settings()

    assert settings.model.image_size % 32 == 0

    assert settings.model.device in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }


def test_logging_level_is_valid() -> None:
    """
    Verify configured logging level.
    """

    settings = Settings()

    assert settings.logging.level in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }


def test_output_directory_not_empty() -> None:
    """
    Verify output directory is configured.
    """

    settings = Settings()

    assert settings.output.directory
    assert len(str(settings.output.directory)) > 0


def test_video_source_not_empty() -> None:
    """
    Verify a video source is configured.
    """

    settings = Settings()

    assert len(str(settings.video.source)) > 0


def test_settings_repr() -> None:
    """
    Verify Settings has a useful string representation.
    """

    settings = Settings()

    representation = repr(settings)

    assert isinstance(representation, str)
    assert "Settings" in representation