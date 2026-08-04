"""
Unit tests for road_accident_detection.config.loader.

These tests verify the configuration loading subsystem. The loader is
responsible for reading configuration files, validating their contents,
constructing Settings objects, and handling configuration-related errors.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from road_accident_detection.config.loader import (
    load_settings,
)
from road_accident_detection.config.settings import Settings
from road_accident_detection.core.exceptions import ConfigurationError


def test_load_settings_returns_settings(
    tmp_path: Path,
) -> None:
    """
    Verify load_settings returns a Settings instance.

    Args:
        tmp_path:
            Temporary directory provided by pytest.
    """

    config = {
        "application": {
            "name": "RoadAccidentAI",
            "version": "1.0.0",
        },
    }

    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert isinstance(settings, Settings)


def test_application_configuration_loaded(
    tmp_path: Path,
) -> None:
    """
    Verify application settings are loaded correctly.
    """

    config = {
        "application": {
            "name": "TestProject",
            "version": "2.0.0",
        },
    }

    path = tmp_path / "config.yaml"

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.application.name == "TestProject"
    assert settings.application.version == "2.0.0"


def test_model_configuration_loaded(
    tmp_path: Path,
) -> None:
    """
    Verify model configuration is loaded correctly.
    """

    config = {
        "model": {
            "detector": "ultralytics",
            "weights": "models/yolo26n.pt",
            "device": "cpu",
            "image_size": 640,
            "confidence": 0.35,
        },
    }

    path = tmp_path / "config.yaml"

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.model.detector == "ultralytics"
    assert settings.model.device == "cpu"
    assert settings.model.image_size == 640
    assert settings.model.confidence == 0.35


def test_video_configuration_loaded(
    tmp_path: Path,
) -> None:
    """
    Verify video configuration is loaded correctly.
    """

    config = {
        "video": {
            "source": "datasets/test.mp4",
            "display": False,
            "loop": True,
        },
    }

    path = tmp_path / "config.yaml"

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.video.source == "datasets/test.mp4"
    assert settings.video.display is False
    assert settings.video.loop is True


def test_output_configuration_loaded(
    tmp_path: Path,
) -> None:
    """
    Verify output configuration is loaded correctly.
    """

    config = {
        "output": {
            "directory": "results",
            "save_video": True,
            "save_frames": True,
        },
    }

    path = tmp_path / "config.yaml"

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.output.directory == "results"
    assert settings.output.save_video is True
    assert settings.output.save_frames is True


def test_missing_configuration_file_raises_error() -> None:
    """
    Verify missing configuration files raise ConfigurationError.
    """

    with pytest.raises(ConfigurationError):
        load_settings(
            Path("does_not_exist.yaml"),
        )


def test_invalid_yaml_raises_error(
    tmp_path: Path,
) -> None:
    """
    Verify malformed YAML raises ConfigurationError.
    """

    config_path = tmp_path / "invalid.yaml"

    config_path.write_text(
        "application: [",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_settings(config_path)


def test_empty_configuration_file(
    tmp_path: Path,
) -> None:
    """
    Verify an empty configuration still produces Settings.
    """

    config_path = tmp_path / "empty.yaml"

    config_path.write_text(
        "",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert isinstance(settings, Settings)


def test_partial_configuration_uses_defaults(
    tmp_path: Path,
) -> None:
    """
    Verify unspecified sections fall back to defaults.
    """

    config = {
        "application": {
            "name": "RoadAccidentAI",
        },
    }

    config_path = tmp_path / "partial.yaml"

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.application.name == "RoadAccidentAI"
    assert settings.model is not None
    assert settings.video is not None
    assert settings.output is not None


def test_multiple_loads_return_independent_objects(
    tmp_path: Path,
) -> None:
    """
    Verify repeated loads return independent Settings instances.
    """

    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        yaml.safe_dump({}),
        encoding="utf-8",
    )

    first = load_settings(config_path)
    second = load_settings(config_path)

    assert first is not second