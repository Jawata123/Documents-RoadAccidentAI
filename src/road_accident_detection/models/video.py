"""
Video domain model for RoadAccidentAI.

This module defines the Video model, which represents metadata about a video
source. It is intentionally independent of OpenCV and any specific video
backend. The model serves as a standardized representation of video
properties throughout the application.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from road_accident_detection.core.types import PathLike

__all__ = [
    "Video",
]


@dataclass(slots=True)
class Video:
    """
    Represents a video source and its metadata.

    This model stores descriptive information about a video source. It does
    not open or process the video itself. Video acquisition is handled by the
    pipeline layer.

    Attributes:
        source:
            Video source path, webcam index, or RTSP/HTTP stream URL.

        name:
            Human-readable name.

        width:
            Frame width in pixels.

        height:
            Frame height in pixels.

        fps:
            Frames per second.

        frame_count:
            Total number of frames if known.

        duration_seconds:
            Total duration in seconds if known.

        is_live:
            Indicates whether the source is a live stream.
    """

    source: PathLike

    name: str = ""

    width: int = 0

    height: int = 0

    fps: float = 0.0

    frame_count: int = 0

    duration_seconds: float = 0.0

    is_live: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Return the video resolution.

        Returns:
            Tuple of (width, height).
        """
        return self.width, self.height

    @property
    def has_known_length(self) -> bool:
        """
        Determine whether the total number of frames is known.

        Returns:
            True if the frame count is greater than zero.
        """
        return self.frame_count > 0

    @property
    def has_known_duration(self) -> bool:
        """
        Determine whether the duration is known.

        Returns:
            True if the duration is greater than zero.
        """
        return self.duration_seconds > 0.0

    @property
    def filename(self) -> str:
        """
        Return the filename if the source is a local file.

        Returns:
            Filename or the original source string for non-file sources.
        """
        source = str(self.source)

        if source.startswith(("rtsp://", "http://", "https://")):
            return source

        return Path(source).name

    @property
    def stem(self) -> str:
        """
        Return the filename without extension.

        Returns:
            Filename stem.
        """
        source = str(self.source)

        if source.startswith(("rtsp://", "http://", "https://")):
            return source

        return Path(source).stem

    @property
    def aspect_ratio(self) -> float:
        """
        Return the aspect ratio.

        Returns:
            Width divided by height.

        Notes:
            Returns 0.0 if the height is zero.
        """
        if self.height == 0:
            return 0.0

        return self.width / self.height

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the video metadata to a serializable dictionary.

        Returns:
            Dictionary representation of the video metadata.
        """
        return {
            "source": str(self.source),
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "is_live": self.is_live,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the video model.
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"source='{self.source}', "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f}, "
            f"live={self.is_live})"
        )