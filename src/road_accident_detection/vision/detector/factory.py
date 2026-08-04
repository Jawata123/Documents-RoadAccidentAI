"""
Detector factory for RoadAccidentAI.

This module implements the Factory Design Pattern for object detector
creation. The factory centralizes detector instantiation so that the rest
of the application never depends on concrete detector implementations.

The factory currently supports the Ultralytics YOLO detector and is designed
to be easily extended with additional detector backends in the future.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

from road_accident_detection.core.exceptions import ConfigurationError

from .base import BaseDetector
from .ultralytics_yolo import UltralyticsYOLODetector

__all__ = [
    "DetectorFactory",
]


class DetectorFactory:
    """
    Factory class for creating detector instances.

    This class provides a centralized mechanism for constructing detector
    implementations based on a detector name.

    Future detector implementations can be added without modifying the
    remainder of the application.

    Example:
        >>> detector = DetectorFactory.create(
        ...     detector_name="ultralytics",
        ...     model_path=Path("models/yolo.pt"),
        ... )
    """

    _SUPPORTED_DETECTORS: dict[str, type[BaseDetector]] = {
        "ultralytics": UltralyticsYOLODetector,
        "yolo": UltralyticsYOLODetector,
    }

    @classmethod
    def create(
        cls,
        detector_name: str,
        model_path: Path,
        device: str = "auto",
    ) -> BaseDetector:
        """
        Create a detector instance.

        Args:
            detector_name:
                Detector backend name.

            model_path:
                Path to the detector weights.

            device:
                Inference device.

        Returns:
            Initialized detector instance.

        Raises:
            ConfigurationError:
                If the detector type is unsupported.
        """

        detector_key = detector_name.strip().lower()

        detector_class = cls._SUPPORTED_DETECTORS.get(detector_key)

        if detector_class is None:
            supported = ", ".join(
                sorted(cls._SUPPORTED_DETECTORS.keys())
            )

            raise ConfigurationError(
                f"Unsupported detector '{detector_name}'. "
                f"Supported detectors: {supported}"
            )

        return detector_class(
            model_path=model_path,
            device=device,
        )

    @classmethod
    def register(
        cls,
        name: str,
        detector_class: type[BaseDetector],
    ) -> None:
        """
        Register a detector implementation.

        This method allows future research modules or plugins to register
        custom detector implementations without modifying the factory.

        Args:
            name:
                Detector name.

            detector_class:
                Detector implementation.
        """

        cls._SUPPORTED_DETECTORS[
            name.strip().lower()
        ] = detector_class

    @classmethod
    def supported_detectors(
        cls,
    ) -> tuple[str, ...]:
        """
        Return the supported detector names.

        Returns:
            Sorted tuple of detector names.
        """

        return tuple(
            sorted(cls._SUPPORTED_DETECTORS.keys())
        )