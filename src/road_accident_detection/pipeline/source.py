"""
Video source implementation for RoadAccidentAI.

This module provides a production-quality wrapper around OpenCV's
VideoCapture. It is responsible for opening video sources, reading frames,
and converting them into the project's Frame model.

The class supports:

- Video files
- USB cameras
- Webcam indices
- RTSP streams
- HTTP streams

The remainder of the application never communicates directly with OpenCV.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import cv2

from road_accident_detection.core.exceptions import (
    FrameReadError,
    VideoOpenError,
)
from road_accident_detection.models.frame import Frame
from road_accident_detection.models.video import Video

__all__ = [
    "VideoSource",
]


class VideoSource:
    """
    Represents an input video source.

    This class wraps OpenCV's VideoCapture while exposing a clean,
    framework-independent interface to the rest of the application.
    """

    def __init__(
        self,
        source: str | int | Path,
    ) -> None:
        """
        Initialize the video source.

        Args:
            source:
                Webcam index, file path, RTSP URL, or HTTP URL.
        """

        self._source = source
        self._capture: cv2.VideoCapture | None = None
        self._frame_index = 0

    @property
    def is_open(self) -> bool:
        """
        Determine whether the video source is open.

        Returns:
            True if opened.
        """

        return (
            self._capture is not None
            and self._capture.isOpened()
        )

    @property
    def frame_index(self) -> int:
        """
        Return the current frame index.

        Returns:
            Frame index.
        """

        return self._frame_index

    def open(self) -> None:
        """
        Open the video source.

        Raises:
            VideoOpenError:
                If the source cannot be opened.
        """

        if self.is_open:
            return

        self._capture = cv2.VideoCapture(self._source)

        if not self._capture.isOpened():
            raise VideoOpenError(
                f"Unable to open video source: {self._source}"
            )

    def close(self) -> None:
        """
        Release the video source.
        """

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self) -> Frame | None:
        """
        Read the next frame.

        Returns:
            Frame if successful.
            None if end of stream.

        Raises:
            FrameReadError:
                If the video source is not open or frame reading fails.
        """

        if not self.is_open:
            raise FrameReadError(
                "Video source is not open."
            )

        assert self._capture is not None

        success, image = self._capture.read()

        if not success:
            return None

        frame = Frame(
            index=self._frame_index,
            image=image,
            fps=self.fps,
            source=str(self._source),
        )

        self._frame_index += 1

        return frame

    @property
    def fps(self) -> float:
        """
        Return the video FPS.

        Returns:
            Frames per second.
        """

        if not self.is_open:
            return 0.0

        assert self._capture is not None

        return float(
            self._capture.get(
                cv2.CAP_PROP_FPS
            )
        )

    @property
    def width(self) -> int:
        """
        Return frame width.

        Returns:
            Width in pixels.
        """

        if not self.is_open:
            return 0

        assert self._capture is not None

        return int(
            self._capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

    @property
    def height(self) -> int:
        """
        Return frame height.

        Returns:
            Height in pixels.
        """

        if not self.is_open:
            return 0

        assert self._capture is not None

        return int(
            self._capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    @property
    def frame_count(self) -> int:
        """
        Return the total number of frames.

        Returns:
            Frame count.
        """

        if not self.is_open:
            return 0

        assert self._capture is not None

        return int(
            self._capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

    def video_metadata(self) -> Video:
        """
        Build the Video domain model.

        Returns:
            Video metadata.
        """

        return Video(
            source=self._source,
            name=Path(str(self._source)).name,
            width=self.width,
            height=self.height,
            fps=self.fps,
            frame_count=self.frame_count,
            duration_seconds=(
                self.frame_count / self.fps
                if self.fps > 0
                else 0.0
            ),
            is_live=isinstance(self._source, int),
        )

    def __enter__(self) -> "VideoSource":
        """
        Enter the runtime context.

        Returns:
            Opened VideoSource.
        """

        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        """
        Exit the runtime context.

        Returns:
            False so exceptions propagate normally.
        """

        self.close()
        return False

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"source='{self._source}', "
            f"open={self.is_open}, "
            f"frame_index={self._frame_index})"
        )