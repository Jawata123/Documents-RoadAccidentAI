"""
Centralized project path management for RoadAccidentAI.

This module provides a single source of truth for all important project
directories and files. It avoids scattered path construction throughout the
codebase and ensures every module works with the same directory structure.

The project root is automatically discovered by locating the directory
containing the ``pyproject.toml`` file.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import (
    CONFIG_DIRECTORY_NAME,
    DATASET_DIRECTORY_NAME,
    DEFAULT_CONFIG_FILENAME,
    LOGGING_CONFIG_FILENAME,
    LOG_DIRECTORY_NAME,
    MODEL_DIRECTORY_NAME,
    OUTPUT_DIRECTORY_NAME,
)

__all__ = [
    "ProjectPaths",
    "get_project_paths",
]


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """
    Represents all important filesystem locations used by the project.

    Attributes:
        project_root:
            Root directory of the repository.

        config_directory:
            Directory containing YAML configuration files.

        dataset_directory:
            Directory containing datasets.

        model_directory:
            Directory containing trained model weights.

        output_directory:
            Directory for generated outputs.

        log_directory:
            Directory where application logs are stored.

        default_config_file:
            Main application configuration.

        logging_config_file:
            Logging configuration file.
    """

    project_root: Path

    config_directory: Path
    dataset_directory: Path
    model_directory: Path
    output_directory: Path
    log_directory: Path

    default_config_file: Path
    logging_config_file: Path

    def create_directories(self) -> None:
        """
        Create all runtime directories if they do not already exist.

        This method is safe to call multiple times.
        """

        self.dataset_directory.mkdir(parents=True, exist_ok=True)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)


def _discover_project_root(start: Path | None = None) -> Path:
    """
    Locate the project root directory.

    The project root is identified as the first parent directory
    containing ``pyproject.toml``.

    Args:
        start:
            Starting directory for the search.
            Defaults to this module's location.

    Returns:
        The repository root.

    Raises:
        RuntimeError:
            If the repository root cannot be located.
    """

    current = (start or Path(__file__)).resolve()

    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").exists():
            return directory

    raise RuntimeError(
        "Unable to locate the project root. "
        "Expected a directory containing 'pyproject.toml'."
    )


def get_project_paths() -> ProjectPaths:
    """
    Build the project's filesystem layout.

    Returns:
        Immutable ProjectPaths instance.
    """

    project_root = _discover_project_root()

    config_directory = project_root / CONFIG_DIRECTORY_NAME
    dataset_directory = project_root / DATASET_DIRECTORY_NAME
    model_directory = project_root / MODEL_DIRECTORY_NAME
    output_directory = project_root / OUTPUT_DIRECTORY_NAME
    log_directory = project_root / LOG_DIRECTORY_NAME

    return ProjectPaths(
        project_root=project_root,
        config_directory=config_directory,
        dataset_directory=dataset_directory,
        model_directory=model_directory,
        output_directory=output_directory,
        log_directory=log_directory,
        default_config_file=(
            config_directory / DEFAULT_CONFIG_FILENAME
        ),
        logging_config_file=(
            config_directory / LOGGING_CONFIG_FILENAME
        ),
    )