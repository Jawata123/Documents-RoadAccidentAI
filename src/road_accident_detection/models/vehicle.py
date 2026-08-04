"""
Vehicle domain model for RoadAccidentAI.

This module defines the Vehicle model used throughout the application. A
Vehicle represents a logical road participant that originates from a
Detection and may later be enriched by future research modules such as
tracking, speed estimation, trajectory analysis, temporal reasoning, and
accident confidence scoring.

The model intentionally contains only generic information required by the
base framework. Future modules can extend this model without breaking the
existing API.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from road_accident_detection.core.types import (
    BoundingBox,
    Confidence,
    ObjectID,
    Point2D,
)

__all__ = [
    "Vehicle",
]


@dataclass(slots=True)
class Vehicle:
    """
    Represents a detected vehicle.

    Attributes:
        object_id:
            Unique object identifier.
            During the base framework this normally matches the detector
            identifier. Future tracking modules will maintain this ID.

        class_id:
            Numerical vehicle class identifier.

        class_name:
            Human-readable class name.

        confidence:
            Detection confidence score.

        bounding_box:
            Bounding box in the format:
            (x_min, y_min, x_max, y_max)

        metadata:
            Additional application-specific metadata.
    """

    object_id: ObjectID

    class_id: int

    class_name: str

    confidence: Confidence

    bounding_box: BoundingBox

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x_min(self) -> float:
        """Return the left coordinate."""
        return self.bounding_box[0]

    @property
    def y_min(self) -> float:
        """Return the top coordinate."""
        return self.bounding_box[1]

    @property
    def x_max(self) -> float:
        """Return the right coordinate."""
        return self.bounding_box[2]

    @property
    def y_max(self) -> float:
        """Return the bottom coordinate."""
        return self.bounding_box[3]

    @property
    def width(self) -> float:
        """
        Return the vehicle bounding-box width.

        Returns:
            Width in pixels.
        """
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """
        Return the vehicle bounding-box height.

        Returns:
            Height in pixels.
        """
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """
        Return the bounding-box area.

        Returns:
            Area in pixels squared.
        """
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        """
        Return the vehicle center.

        Returns:
            Tuple containing (x, y).
        """
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    def update_bounding_box(
        self,
        bounding_box: BoundingBox,
        confidence: Confidence,
    ) -> None:
        """
        Update the vehicle with a new detection.

        This method will be used by future tracking modules to refresh the
        vehicle state after each frame.

        Args:
            bounding_box:
                Updated bounding box.

            confidence:
                Updated confidence score.
        """

        self.bounding_box = bounding_box
        self.confidence = confidence

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the vehicle to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the vehicle.
        """

        return {
            "object_id": self.object_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bounding_box": list(self.bounding_box),
            "center": list(self.center),
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the vehicle.
        """

        return (
            f"{self.__class__.__name__}("
            f"id={self.object_id}, "
            f"class='{self.class_name}', "
            f"confidence={self.confidence:.3f}, "
            f"bbox={self.bounding_box})"
        )