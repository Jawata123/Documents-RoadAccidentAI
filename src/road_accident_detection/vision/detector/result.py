"""
Detector result models for RoadAccidentAI.

This module defines the standardized output returned by every detector
implementation. Regardless of whether detections originate from
Ultralytics YOLO or a future detector, they are converted into this common
representation before being passed to the rest of the application.

The detector result is intentionally independent of any third-party
framework and contains only project domain models.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from road_accident_detection.models.detection import Detection

__all__ = [
    "DetectionResult",
]


@dataclass(slots=True)
class DetectionResult:
    """
    Represents the output of a detector for a single frame.

    Attributes:
        detections:
            List of detected objects.

        inference_time_ms:
            Model inference time in milliseconds.

        frame_index:
            Index of the processed frame.

        metadata:
            Optional detector-specific metadata.
    """

    detections: list[Detection] = field(default_factory=list)

    inference_time_ms: float = 0.0

    frame_index: int = -1

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """
        Return the number of detections.

        Returns:
            Number of detections.
        """
        return len(self.detections)

    @property
    def is_empty(self) -> bool:
        """
        Determine whether any detections exist.

        Returns:
            True if no detections are present.
        """
        return not self.detections

    def add_detection(
        self,
        detection: Detection,
    ) -> None:
        """
        Add a detection.

        Args:
            detection:
                Detection to add.
        """
        self.detections.append(detection)

    def extend(
        self,
        detections: list[Detection],
    ) -> None:
        """
        Append multiple detections.

        Args:
            detections:
                Detection list.
        """
        self.detections.extend(detections)

    def clear(self) -> None:
        """
        Remove all detections.
        """
        self.detections.clear()

    def filter_by_class(
        self,
        class_name: str,
    ) -> list[Detection]:
        """
        Return detections belonging to a class.

        Args:
            class_name:
                Target class name.

        Returns:
            Matching detections.
        """
        return [
            detection
            for detection in self.detections
            if detection.class_name == class_name
        ]

    def filter_by_confidence(
        self,
        minimum_confidence: float,
    ) -> list[Detection]:
        """
        Return detections above a confidence threshold.

        Args:
            minimum_confidence:
                Minimum confidence.

        Returns:
            Matching detections.
        """
        return [
            detection
            for detection in self.detections
            if detection.confidence >= minimum_confidence
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the detection result into a serializable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "frame_index": self.frame_index,
            "inference_time_ms": self.inference_time_ms,
            "count": self.count,
            "detections": [
                detection.to_dict()
                for detection in self.detections
            ],
            "metadata": self.metadata,
        }

    def __iter__(self):
        """
        Iterate over detections.

        Returns:
            Detection iterator.
        """
        return iter(self.detections)

    def __len__(self) -> int:
        """
        Return the number of detections.

        Returns:
            Number of detections.
        """
        return self.count

    def __getitem__(
        self,
        index: int,
    ) -> Detection:
        """
        Retrieve a detection by index.

        Args:
            index:
                Detection index.

        Returns:
            Detection instance.
        """
        return self.detections[index]

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"count={self.count}, "
            f"inference_time_ms={self.inference_time_ms:.2f}, "
            f"frame_index={self.frame_index})"
        )