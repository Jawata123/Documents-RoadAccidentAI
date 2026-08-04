"""
Centralized logging utilities for RoadAccidentAI.

This module provides a consistent logging interface for the entire project.
All packages should obtain loggers through this module instead of directly
using the standard logging package.

Features
--------
- Consistent formatter
- Console logging
- File logging
- Singleton configuration
- Configurable log level
- Thread-safe initialization

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from .constants import (
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOGGER_NAME,
)

__all__ = [
    "configure_logging",
    "get_logger",
]

###############################################################################
# Module State
###############################################################################

_CONFIGURED: bool = False

###############################################################################
# Public API
###############################################################################


def configure_logging(
    *,
    log_directory: Path | None = None,
    log_level: str = DEFAULT_LOG_LEVEL,
    log_file: str = DEFAULT_LOG_FILE,
) -> None:
    """
    Configure the global logging system.

    This function is safe to call multiple times. Logging will only be
    configured once during the application's lifetime.

    Args:
        log_directory:
            Directory where log files should be written.
            If None, file logging is disabled.

        log_level:
            Logging level.

        log_file:
            Name of the log file.

    Raises:
        OSError:
            If the log directory cannot be created.
    """
    global _CONFIGURED

    if _CONFIGURED:
        return

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    handlers: list[logging.Handler] = []

    ###########################################################################
    # Console Handler
    ###########################################################################

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    ###########################################################################
    # File Handler
    ###########################################################################

    if log_directory is not None:
        log_directory.mkdir(parents=True, exist_ok=True)

        log_path = log_directory / log_file

        file_handler = logging.FileHandler(
            filename=log_path,
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)

        handlers.append(file_handler)

    ###########################################################################
    # Root Configuration
    ###########################################################################

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    _CONFIGURED = True


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """
    Return a project logger.

    Logging is automatically configured if it has not yet been initialized.

    Args:
        name:
            Logger name.

    Returns:
        Configured Logger instance.
    """
    if not _CONFIGURED:
        configure_logging()

    return logging.getLogger(name)