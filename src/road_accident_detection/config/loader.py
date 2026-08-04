"""
Configuration loader for RoadAccidentAI.

This module is responsible for loading the application's YAML configuration
file, validating it against the project's configuration schema, and returning
an immutable Settings instance.

The loader is intentionally the only component allowed to read configuration
files from disk. All other modules should obtain configuration through the
public functions defined here.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from road_accident_detection.core.exceptions import (
    ConfigurationError,
)
from road_accident_detection.core.paths import get_project_paths
from road_accident_detection.core.validation import (
    validate_file_exists,
)

from .schemas import ApplicationConfig
from .settings import Settings

__all__ = [
    "load_settings",
]


def _read_yaml(path: Path) -> dict[str, Any]:
    """
    Read a YAML configuration file.

    Args:
        path:
            Path to the YAML configuration file.

    Returns:
        Parsed YAML dictionary.

    Raises:
        ConfigurationError:
            If the YAML file cannot be read or does not contain
            a dictionary at the root level.
    """

    try:
        with path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file) or {}

    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML configuration: {path}"
        ) from exc

    except OSError as exc:
        raise ConfigurationError(
            f"Unable to read configuration file: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            "The root of the configuration file must be a mapping."
        )

    return data


def load_settings(
    config_file: Path | None = None,
) -> Settings:
    """
    Load and validate application settings.

    Args:
        config_file:
            Optional path to a configuration file.
            If omitted, the project's default configuration file
            is loaded.

    Returns:
        Immutable Settings instance.

    Raises:
        ConfigurationError:
            If configuration loading or validation fails.
    """

    paths = get_project_paths()

    configuration_path = (
        config_file
        if config_file is not None
        else paths.default_config_file
    )

    configuration_path = validate_file_exists(configuration_path)

    configuration_data = _read_yaml(configuration_path)

    try:
        validated = ApplicationConfig.model_validate(
            configuration_data
        )

    except PydanticValidationError as exc:
        raise ConfigurationError(
            f"Configuration validation failed:\n{exc}"
        ) from exc

    return Settings.model_validate(
        validated.model_dump()
    )