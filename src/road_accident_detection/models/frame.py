"""
Frame domain model for RoadAccidentAI.

This module defines the Frame model used throughout the project to represent
a single video frame and its associated metadata.

The model intentionally does not depend on OpenCV, Ultralytics, or any
tracking implementation. It serves as a lightweight, immutable container
passed between pipeline stages.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from road_accident_detection.core.types import FrameIndex, Size2D

__all__ = [
    "Frame",
]


@dataclass(slots=True)
class Frame:
    """
    Represents a single video frame.

    A Frame contains the raw image data together with metadata describing
    where the frame came from and when it was processed.

    Attributes:
        index:
            Zero-based frame number.

        image:
            Frame image stored as a NumPy array.

        timestamp:
            Capture or processing timestamp.

        fps:
            Frames per second of the source video.

        source:
            Human-readable source description.

    Example:
        >>> frame = Frame(
        ...     index=25,
        ...     image=image,
        ...     fps=30.0,
        ...     source="traffic_camera_01",
        ... )
    """

    index: FrameIndex

    image: NDArray[np.uint8]

    timestamp: datetime = field(
        default_factory=datetime.utcnow
    )

    fps: float = 0.0

    source: str = ""

    @property
    def width(self) -> int:
        """
        Return the frame width.

        Returns:
            Width in pixels.
        """
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """
        Return the frame height.

        Returns:
            Height in pixels.
        """
        return int(self.image.shape[0])

    @property
    def channels(self) -> int:
        """
        Return the number of image channels.

        Returns:
            Number of channels.

        Notes:
            Returns 1 for grayscale images.
        """
        if self.image.ndim == 2:
            return 1

        return int(self.image.shape[2])

    @property
    def size(self) -> Size2D:
        """
        Return the frame size.

        Returns:
            Tuple of (width, height).
        """
        return (
            self.width,
            self.height,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the NumPy image shape.

        Returns:
            Image shape.
        """
        return self.image.shape

    @property
    def is_color(self) -> bool:
        """
        Determine whether the frame is a color image.

        Returns:
            True if the image contains multiple channels.
        """
        return self.channels >= 3

    def copy(self) -> "Frame":
        """
        Create a deep copy of the frame.

        Returns:
            New Frame instance containing a copied image.
        """
        return Frame(
            index=self.index,
            image=self.image.copy(),
            timestamp=self.timestamp,
            fps=self.fps,
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metadata to a serializable dictionary.

        The raw image is intentionally omitted because NumPy arrays are
        not JSON serializable.

        Returns:
            Dictionary containing frame metadata.
        """
        return {
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "fps": self.fps,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }

    def __len__(self) -> int:
        """
        Return the total number of pixels.

        Returns:
            Number of pixels.
        """
        return self.width * self.height

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the frame.
        """
        return (
            f"{self.__class__.__name__}("
            f"index={self.index}, "
            f"size={self.width}x{self.height}, "
            f"channels={self.channels}, "
            f"fps={self.fps:.2f})"
        )