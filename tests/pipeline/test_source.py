"""
Unit tests for road_accident_detection.pipeline.source.

These tests verify the VideoSource pipeline component responsible for
opening videos, webcams, and image sequences while exposing a consistent
frame-reading interface to the processing pipeline.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from road_accident_detection.pipeline.source import VideoSource


def test_video_source_initialization() -> None:
    """
    Verify VideoSource initialization.
    """

    source = VideoSource(
        source="datasets/video.mp4",
    )

    assert source is not None
    assert source.source == "datasets/video.mp4"


def test_video_source_string_path() -> None:
    """
    Verify string source paths are accepted.
    """

    source = VideoSource(
        source="datasets/demo.mp4",
    )

    assert isinstance(source.source, str)


def test_video_source_path_object() -> None:
    """
    Verify pathlib.Path objects are accepted.
    """

    source = VideoSource(
        source=Path("datasets/demo.mp4"),
    )

    assert isinstance(source.source, Path)


def test_video_source_webcam_index() -> None:
    """
    Verify webcam indices are accepted.
    """

    source = VideoSource(
        source=0,
    )

    assert source.source == 0


@pytest.mark.parametrize(
    "source",
    [
        "datasets/a.mp4",
        "datasets/b.avi",
        "datasets/c.mov",
        Path("datasets/d.mkv"),
        0,
        1,
    ],
)
def test_supported_source_types(
    source: str | Path | int,
) -> None:
    """
    Verify supported source types.

    Args:
        source:
            Video source under test.
    """

    video_source = VideoSource(
        source=source,
    )

    assert video_source.source == source


def test_video_source_closed_initially() -> None:
    """
    Verify newly created source is not opened.
    """

    source = VideoSource(
        source="datasets/video.mp4",
    )

    assert source.is_open is False


def test_video_source_has_open_method() -> None:
    """
    Verify open() exists.
    """

    source = VideoSource(
        source="datasets/video.mp4",
    )

    assert callable(source.open)


def test_video_source_has_close_method() -> None:
    """
    Verify close() exists.
    """

    source = VideoSource(
        source="datasets/video.mp4",
    )

    assert callable(source.close)


def test_video_source_has_read_method() -> None:
    """
    Verify read() exists.
    """

    source = VideoSource(
        source="datasets/video.mp4",
    )

    assert callable(source.read)


def test_video_source_has_reset_method() -> None:
    """
    Verify reset() exists.
    """

    source = VideoSource(
        source="datasets/video.mp4",
    )

    assert callable(source.reset)


def test_video_source_repr() -> None:
    """
    Verify string representation.
    """

    source = VideoSource(
        source="datasets/video.mp4",
    )

    representation = repr(source)

    assert isinstance(representation, str)
    assert "VideoSource" in representation


def test_multiple_video_sources_are_independent() -> None:
    """
    Verify multiple instances are independent.
    """

    first = VideoSource(
        source="datasets/a.mp4",
    )

    second = VideoSource(
        source="datasets/b.mp4",
    )

    assert first is not second
    assert first.source != second.source


def test_video_source_source_property() -> None:
    """
    Verify source property is preserved.
    """

    source = VideoSource(
        source="datasets/sample.mp4",
    )

    assert source.source == "datasets/sample.mp4"


def test_video_source_webcam_property() -> None:
    """
    Verify webcam sources remain integers.
    """

    source = VideoSource(
        source=1,
    )

    assert isinstance(source.source, int)