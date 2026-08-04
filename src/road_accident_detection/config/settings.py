"""
Application settings interface for RoadAccidentAI.

This module provides the immutable runtime settings object used throughout
the application. Settings are loaded from YAML configuration files by the
configuration loader and validated using the Pydantic schemas defined in
``schemas.py``.

This module intentionally contains no file I/O. Configuration loading is
handled exclusively by ``loader.py``.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pydantic import ConfigDict

from .schemas import ApplicationConfig

__all__ = [
    "Settings",
]


class Settings(ApplicationConfig):
    """
    Immutable runtime application settings.

    This class represents the fully validated configuration used by the
    application during execution. It extends the base ``ApplicationConfig``
    schema to provide a dedicated runtime settings type while preserving
    the same validation rules.

    The class is intentionally lightweight so future configuration sections
    (tracking, trajectory analysis, REST API, benchmarking, etc.) can be
    added without changing how settings are consumed by the rest of the
    project.

    Example:
        >>> from road_accident_detection.config.loader import load_settings
        >>> settings = load_settings()
        >>> print(settings.model.weights)
        >>> print(settings.video.display)
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the settings object.
        """
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model.weights}', "
            f"device='{self.model.device}', "
            f"video='{self.video.source}')"
        )