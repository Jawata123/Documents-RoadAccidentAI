"""
Detection domain model for RoadAccidentAI.

This module defines the Detection model, which represents a single object
detected within a video frame. The model is intentionally independent of
Ultralytics YOLO, OpenCV, or any future object detector.

The Detection model acts as the standardized data exchange format between
the detector, tracker, feature extraction, event reasoning, and evaluation
modules.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from road_accident_detection.core.types import (
    BoundingBox,
    Confidence,
    Point2D,
)

__all__ = [
    "Detection",
]


@dataclass(slots=True)
class Detection:
    """
    Represents a single detected object.

    Attributes:
        class_id:
            Numeric class identifier.

        class_name:
            Human-readable class label.

        confidence:
            Detection confidence score.

        bounding_box:
            Bounding box represented as
            (x_min, y_min, x_max, y_max).

        tracker_id:
            Object tracking ID assigned by a future tracking module.
            None until tracking is performed.
    """

    class_id: int

    class_name: str

    confidence: Confidence

    bounding_box: BoundingBox

    tracker_id: int | None = None

    @property
    def x_min(self) -> float:
        """Left coordinate."""
        return self.bounding_box[0]

    @property
    def y_min(self) -> float:
        """Top coordinate."""
        return self.bounding_box[1]

    @property
    def x_max(self) -> float:
        """Right coordinate."""
        return self.bounding_box[2]

    @property
    def y_max(self) -> float:
        """Bottom coordinate."""
        return self.bounding_box[3]

    @property
    def width(self) -> float:
        """
        Width of the bounding box.

        Returns:
            Bounding box width.
        """
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """
        Height of the bounding box.

        Returns:
            Bounding box height.
        """
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """
        Bounding box area.

        Returns:
            Area in pixels squared.
        """
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        """
        Center point of the bounding box.

        Returns:
            Tuple of (x, y).
        """
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    @property
    def is_tracked(self) -> bool:
        """
        Determine whether this detection has been assigned a tracker ID.

        Returns:
            True if a tracker ID exists.
        """
        return self.tracker_id is not None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the detection to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the detection.
        """
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bounding_box": list(self.bounding_box),
            "tracker_id": self.tracker_id,
            "center": list(self.center),
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the detection.
        """
        return (
            f"{self.__class__.__name__}("
            f"class='{self.class_name}', "
            f"confidence={self.confidence:.3f}, "
            f"bbox={self.bounding_box}, "
            f"tracker_id={self.tracker_id})"
        )