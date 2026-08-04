"""
Pipeline runner for RoadAccidentAI.

This module provides the main execution engine for the RoadAccidentAI
framework. The runner coordinates the video source, frame processor, and
detector while remaining independent of any specific object detection
framework.

The runner is intentionally designed to serve as the foundation for future
research modules including:

- Multi-object tracking
- Speed estimation
- Trajectory analysis
- Temporal reasoning
- Accident confidence scoring
- Event verification
- Alert generation

These features are NOT implemented here. The runner simply provides the
processing loop into which future stages can be integrated.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from road_accident_detection.models.frame import Frame
from road_accident_detection.pipeline.frame import FrameProcessor
from road_accident_detection.pipeline.source import VideoSource
from road_accident_detection.vision.detector.result import DetectionResult

__all__ = [
    "PipelineRunner",
]


@dataclass(slots=True)
class PipelineRunner:
    """
    Coordinates the execution of the video processing pipeline.

    Attributes:
        source:
            Video source.

        processor:
            Frame processor.
    """

    source: VideoSource

    processor: FrameProcessor

    def run(self) -> None:
        """
        Execute the processing pipeline.

        The method continuously reads frames from the configured
        VideoSource and processes each frame until the video ends.

        Raises:
            Any exception raised by the underlying source or detector.
        """

        self.source.open()

        try:
            while True:
                frame = self.source.read()

                if frame is None:
                    break

                self.process_frame(frame)

        finally:
            self.source.close()

    def process_frame(
        self,
        frame: Frame,
    ) -> DetectionResult:
        """
        Process a single frame.

        This method represents the primary extension point for future
        research modules. Future pipeline stages such as tracking,
        trajectory analysis, speed estimation, and accident reasoning
        can be inserted here without changing the overall architecture.

        Args:
            frame:
                Input frame.

        Returns:
            Detection result produced by the detector.
        """

        detection_result = self.processor.process(frame)

        return detection_result

    def __call__(self) -> None:
        """
        Execute the pipeline.

        Allows the runner to be called like a function.

        Example:
            >>> runner()
        """

        self.run()

    def __repr__(self) -> str:
        """
        Return a concise developer-friendly representation.

        Returns:
            String representation of the runner.
        """

        return (
            f"{self.__class__.__name__}("
            f"source={self.source}, "
            f"processor={self.processor})"
        )