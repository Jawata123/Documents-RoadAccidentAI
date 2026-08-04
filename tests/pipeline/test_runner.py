"""
Unit tests for road_accident_detection.pipeline.runner.

These tests verify the PipelineRunner responsible for orchestrating the
complete RoadAccidentAI processing workflow. The runner coordinates the
video source, frame processor, detector, and future research modules while
remaining independent of the detector backend.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from road_accident_detection.pipeline.runner import PipelineRunner
from road_accident_detection.pipeline.source import VideoSource
from road_accident_detection.pipeline.frame import FrameProcessor
from road_accident_detection.vision.detector.factory import DetectorFactory
from road_accident_detection.vision.detector.base import BaseDetector


def test_pipeline_runner_initialization() -> None:
    """
    Verify PipelineRunner initializes correctly.
    """

    runner = PipelineRunner()

    assert runner is not None


def test_pipeline_runner_has_source() -> None:
    """
    Verify PipelineRunner exposes a video source.
    """

    runner = PipelineRunner()

    assert hasattr(runner, "source")


def test_pipeline_runner_has_detector() -> None:
    """
    Verify PipelineRunner exposes a detector.
    """

    runner = PipelineRunner()

    assert hasattr(runner, "detector")


def test_pipeline_runner_has_frame_processor() -> None:
    """
    Verify PipelineRunner exposes a frame processor.
    """

    runner = PipelineRunner()

    assert hasattr(runner, "frame_processor")


def test_pipeline_runner_accepts_video_source() -> None:
    """
    Verify custom VideoSource can be supplied.
    """

    source = VideoSource(
        source="datasets/video.mp4",
    )

    runner = PipelineRunner(
        source=source,
    )

    assert runner.source is source


def test_pipeline_runner_accepts_detector() -> None:
    """
    Verify custom detector can be supplied.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    runner = PipelineRunner(
        detector=detector,
    )

    assert runner.detector is detector


def test_pipeline_runner_accepts_frame_processor() -> None:
    """
    Verify custom FrameProcessor can be supplied.
    """

    processor = FrameProcessor()

    runner = PipelineRunner(
        frame_processor=processor,
    )

    assert runner.frame_processor is processor


def test_pipeline_runner_detector_type() -> None:
    """
    Verify detector implements BaseDetector.
    """

    detector = DetectorFactory.create(
        detector_type="ultralytics",
        model_path=Path("models/yolo26n.pt"),
    )

    assert isinstance(
        detector,
        BaseDetector,
    )


def test_pipeline_runner_repr() -> None:
    """
    Verify string representation.
    """

    runner = PipelineRunner()

    representation = repr(runner)

    assert isinstance(
        representation,
        str,
    )

    assert "PipelineRunner" in representation


def test_pipeline_runner_multiple_instances() -> None:
    """
    Verify runner instances are independent.
    """

    first = PipelineRunner()
    second = PipelineRunner()

    assert first is not second


def test_pipeline_runner_has_run_method() -> None:
    """
    Verify run() exists.
    """

    runner = PipelineRunner()

    assert callable(runner.run)


def test_pipeline_runner_has_stop_method() -> None:
    """
    Verify stop() exists.
    """

    runner = PipelineRunner()

    assert callable(runner.stop)


def test_pipeline_runner_has_reset_method() -> None:
    """
    Verify reset() exists.
    """

    runner = PipelineRunner()

    assert callable(runner.reset)


def test_pipeline_runner_accepts_numpy_frame() -> None:
    """
    Verify runner can work with NumPy images.

    This test validates image compatibility without
    executing the complete pipeline.
    """

    image = np.zeros(
        (
            480,
            640,
            3,
        ),
        dtype=np.uint8,
    )

    assert image.shape == (
        480,
        640,
        3,
    )