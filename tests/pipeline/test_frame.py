"""
Unit tests for road_accident_detection.pipeline.frame.

These tests verify the FrameProcessor pipeline component responsible for
preparing image frames before they are passed to the detector. The tests
ensure correct processing behavior, interface compliance, and object
independence.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import numpy as np

from road_accident_detection.models.frame import Frame
from road_accident_detection.pipeline.frame import FrameProcessor


def create_test_image() -> np.ndarray:
    """
    Create a reusable RGB test image.

    Returns:
        NumPy RGB image.
    """

    return np.zeros(
        (
            480,
            640,
            3,
        ),
        dtype=np.uint8,
    )


def create_frame() -> Frame:
    """
    Create a reusable Frame object.

    Returns:
        Frame instance.
    """

    return Frame(
        index=1,
        image=create_test_image(),
        fps=30.0,
        source="video.mp4",
    )


def test_frame_processor_initialization() -> None:
    """
    Verify FrameProcessor initialization.
    """

    processor = FrameProcessor()

    assert processor is not None


def test_process_returns_frame() -> None:
    """
    Verify process() returns a Frame.
    """

    processor = FrameProcessor()

    frame = create_frame()

    result = processor.process(frame)

    assert isinstance(result, Frame)


def test_process_preserves_index() -> None:
    """
    Verify frame index is preserved.
    """

    processor = FrameProcessor()

    frame = create_frame()

    result = processor.process(frame)

    assert result.index == frame.index


def test_process_preserves_image_shape() -> None:
    """
    Verify image dimensions remain unchanged.
    """

    processor = FrameProcessor()

    frame = create_frame()

    result = processor.process(frame)

    assert result.image.shape == (
        480,
        640,
        3,
    )


def test_process_preserves_dtype() -> None:
    """
    Verify image dtype remains unchanged.
    """

    processor = FrameProcessor()

    frame = create_frame()

    result = processor.process(frame)

    assert result.image.dtype == np.uint8


def test_process_preserves_source() -> None:
    """
    Verify source metadata is preserved.
    """

    processor = FrameProcessor()

    frame = create_frame()

    result = processor.process(frame)

    assert result.source == frame.source


def test_process_preserves_fps() -> None:
    """
    Verify FPS metadata is preserved.
    """

    processor = FrameProcessor()

    frame = create_frame()

    result = processor.process(frame)

    assert result.fps == frame.fps


def test_multiple_process_calls() -> None:
    """
    Verify processor supports repeated calls.
    """

    processor = FrameProcessor()

    frame = create_frame()

    first = processor.process(frame)
    second = processor.process(frame)

    assert first is not second


def test_process_accepts_numpy_image_frame() -> None:
    """
    Verify processed frame contains a NumPy image.
    """

    processor = FrameProcessor()

    frame = create_frame()

    result = processor.process(frame)

    assert isinstance(
        result.image,
        np.ndarray,
    )


def test_frame_processor_repr() -> None:
    """
    Verify string representation.
    """

    processor = FrameProcessor()

    representation = repr(processor)

    assert isinstance(
        representation,
        str,
    )

    assert "FrameProcessor" in representation


def test_processor_independent_instances() -> None:
    """
    Verify processor instances are independent.
    """

    first = FrameProcessor()
    second = FrameProcessor()

    assert first is not second


def test_processed_frame_dimensions() -> None:
    """
    Verify processed frame dimensions.
    """

    processor = FrameProcessor()

    frame = create_frame()

    result = processor.process(frame)

    assert result.width == 640
    assert result.height == 480