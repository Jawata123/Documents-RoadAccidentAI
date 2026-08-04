"""
Abstract detector interface for RoadAccidentAI.

This module defines the abstract base class that every object detector must
implement. The purpose of this abstraction is to isolate the remainder of
the application from any specific object detection framework.

All detector implementations (Ultralytics YOLO, RT-DETR, Detectron2,
Grounding DINO, etc.) must inherit from BaseDetector and implement the
required interface.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from road_accident_detection.models.frame import Frame

from .result import DetectionResult

__all__ = [
    "BaseDetector",
]


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.

    Concrete detector implementations must inherit from this class and
    implement every abstract method.

    This interface ensures that the processing pipeline depends only on
    detector abstractions rather than framework-specific implementations.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
    ) -> None:
        """
        Initialize the detector.

        Args:
            model_path:
                Path to the model weights.

            device:
                Target inference device.
        """
        self._model_path = Path(model_path)
        self._device = device

    @property
    def model_path(self) -> Path:
        """
        Return the model path.

        Returns:
            Path to the model weights.
        """
        return self._model_path

    @property
    def device(self) -> str:
        """
        Return the inference device.

        Returns:
            Device name.
        """
        return self._device

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the detector name.

        Returns:
            Human-readable detector name.
        """

    @property
    @abstractmethod
    def version(self) -> str:
        """
        Return the detector version.

        Returns:
            Detector version string.
        """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Determine whether the model is loaded.

        Returns:
            True if the detector is ready for inference.
        """

    @abstractmethod
    def load(self) -> None:
        """
        Load the detection model.

        Raises:
            ModelLoadError:
                If the model cannot be loaded.
        """

    @abstractmethod
    def unload(self) -> None:
        """
        Release detector resources.
        """

    @abstractmethod
    def detect(
        self,
        frame: Frame,
    ) -> DetectionResult:
        """
        Run inference on a frame.

        Args:
            frame:
                Input frame.

        Returns:
            DetectionResult containing standardized detections.
        """

    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the detector.

        This method performs any initialization or dummy inference required
        before real-time processing begins.
        """

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """
        Return detector metadata.

        Returns:
            Dictionary describing the detector.
        """

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the detector.
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"device='{self.device}', "
            f"loaded={self.is_loaded})"
        )