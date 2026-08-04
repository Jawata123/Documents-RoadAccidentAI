"""
Frame processing utilities for RoadAccidentAI.

This module provides reusable frame-level processing functions used by the
pipeline. The processor is responsible for validating incoming frames and
delegating object detection to the configured detector. It deliberately does
not perform tracking, accident reasoning, or any research-specific logic.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from road_accident_detection.models.frame import Frame
from road_accident_detection.vision.detector.base import BaseDetector
from road_accident_detection.vision.detector.result import DetectionResult

__all__ = [
    "FrameProcessor",
]


@dataclass(slots=True)
class FrameProcessor:
    """
    Frame processing pipeline.

    This class performs frame-level processing by invoking the configured
    detector and returning a standardized DetectionResult.

    Attributes:
        detector:
            Object detector implementation.
    """

    detector: BaseDetector

    def process(
        self,
        frame: Frame,
    ) -> DetectionResult:
        """
        Process a single frame.

        Args:
            frame:
                Input frame.

        Returns:
            DetectionResult produced by the detector.

        Raises:
            ValueError:
                If the supplied frame is invalid.
        """

        if frame.image is None:
            raise ValueError(
                "Frame image cannot be None."
            )

        if frame.image.size == 0:
            raise ValueError(
                "Frame image is empty."
            )

        return self.detector.detect(frame)

    def __call__(
        self,
        frame: Frame,
    ) -> DetectionResult:
        """
        Allow the processor instance to be called like a function.

        Args:
            frame:
                Input frame.

        Returns:
            DetectionResult.
        """

        return self.process(frame)

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the processor.
        """

        return (
            f"{self.__class__.__name__}("
            f"detector={self.detector.name})"
        )