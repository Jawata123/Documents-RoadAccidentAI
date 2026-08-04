"""
Default configuration values for RoadAccidentAI.

This module defines immutable default configuration values used throughout
the project. These values serve as safe fallbacks whenever configuration
entries are not explicitly provided by the user.

IMPORTANT:
-----------
This module should contain only generic application defaults.

Research parameters such as:
    - confidence thresholds
    - IoU thresholds
    - tracking parameters
    - speed thresholds
    - accident decision logic

must be stored in configuration files (configs/default.yaml),
NOT inside this module.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from road_accident_detection.core.constants import (
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_WINDOW_NAME,
)

__all__ = [
    "ApplicationDefaults",
]


@dataclass(frozen=True, slots=True)
class ApplicationDefaults:
    """
    Immutable application-wide default settings.

    These values represent safe defaults for the framework itself and are
    independent of any specific accident detection experiment.

    Attributes:
        log_level:
            Default logging level.

        log_file:
            Default log filename.

        create_output_directory:
            Whether runtime output directories should be created
            automatically.

        save_logs:
            Whether log files should be written to disk.

        save_output_video:
            Whether processed videos should be saved by default.

        display_output:
            Whether processed frames should be displayed.

        recursive_directory_search:
            Whether dataset directory searches are recursive.

        window_name:
            Default OpenCV display window title.

        supported_configuration_extensions:
            Supported configuration file extensions.
    """

    log_level: str = DEFAULT_LOG_LEVEL

    log_file: str = DEFAULT_LOG_FILE

    create_output_directory: bool = True

    save_logs: bool = True

    save_output_video: bool = False

    display_output: bool = True

    recursive_directory_search: bool = True

    window_name: str = DEFAULT_WINDOW_NAME

    supported_configuration_extensions: tuple[str, ...] = (
        ".yaml",
        ".yml",
    )


###############################################################################
# Singleton Defaults
###############################################################################

DEFAULTS: Final[ApplicationDefaults] = ApplicationDefaults()