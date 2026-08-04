"""
Unit tests for road_accident_detection.models.video.

These tests verify the Video domain model, ensuring correct initialization,
property behavior, serialization, and object independence.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from road_accident_detection.models.video import Video


def test_video_initialization() -> None:
    """
    Verify Video initialization.
    """

    video = Video(
        source="datasets/test.mp4",
        name="test.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=900,
        duration_seconds=30.0,
        is_live=False,
    )

    assert video.source == "datasets/test.mp4"
    assert video.name == "test.mp4"
    assert video.width == 1920
    assert video.height == 1080
    assert video.fps == 30.0
    assert video.frame_count == 900
    assert video.duration_seconds == 30.0
    assert video.is_live is False


def test_video_resolution() -> None:
    """
    Verify resolution property.
    """

    video = Video(
        source="video.mp4",
        name="video.mp4",
        width=1280,
        height=720,
        fps=25.0,
        frame_count=100,
        duration_seconds=4.0,
        is_live=False,
    )

    assert video.resolution == (1280, 720)


def test_video_aspect_ratio() -> None:
    """
    Verify aspect ratio calculation.
    """

    video = Video(
        source="video.mp4",
        name="video.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=100,
        duration_seconds=3.33,
        is_live=False,
    )

    assert abs(video.aspect_ratio - (1920 / 1080)) < 1e-6


def test_video_total_pixels() -> None:
    """
    Verify total pixel calculation.
    """

    video = Video(
        source="video.mp4",
        name="video.mp4",
        width=640,
        height=480,
        fps=30.0,
        frame_count=300,
        duration_seconds=10.0,
        is_live=False,
    )

    assert video.width * video.height == 307200


def test_video_to_dict() -> None:
    """
    Verify dictionary serialization.
    """

    video = Video(
        source="datasets/sample.mp4",
        name="sample.mp4",
        width=640,
        height=480,
        fps=24.0,
        frame_count=240,
        duration_seconds=10.0,
        is_live=False,
    )

    data = video.to_dict()

    assert data["source"] == "datasets/sample.mp4"
    assert data["name"] == "sample.mp4"
    assert data["width"] == 640
    assert data["height"] == 480
    assert data["fps"] == 24.0
    assert data["frame_count"] == 240
    assert data["duration_seconds"] == 10.0
    assert data["is_live"] is False


def test_video_repr() -> None:
    """
    Verify string representation.
    """

    video = Video(
        source="datasets/video.mp4",
        name="video.mp4",
        width=1280,
        height=720,
        fps=60.0,
        frame_count=600,
        duration_seconds=10.0,
        is_live=False,
    )

    representation = repr(video)

    assert isinstance(representation, str)
    assert "Video" in representation


def test_live_video_flag() -> None:
    """
    Verify live video flag.
    """

    stream = Video(
        source=0,
        name="Webcam",
        width=1280,
        height=720,
        fps=30.0,
        frame_count=0,
        duration_seconds=0.0,
        is_live=True,
    )

    assert stream.is_live is True


def test_video_duration_positive() -> None:
    """
    Verify duration is positive.
    """

    video = Video(
        source="video.mp4",
        name="video.mp4",
        width=640,
        height=480,
        fps=25.0,
        frame_count=250,
        duration_seconds=10.0,
        is_live=False,
    )

    assert video.duration_seconds > 0


def test_multiple_video_instances_are_independent() -> None:
    """
    Verify Video instances are independent.
    """

    first = Video(
        source="a.mp4",
        name="a.mp4",
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        duration_seconds=3.33,
        is_live=False,
    )

    second = Video(
        source="b.mp4",
        name="b.mp4",
        width=1920,
        height=1080,
        fps=60.0,
        frame_count=600,
        duration_seconds=10.0,
        is_live=False,
    )

    assert first is not second
    assert first.source != second.source


def test_video_fps_positive() -> None:
    """
    Verify FPS is positive.
    """

    video = Video(
        source="video.mp4",
        name="video.mp4",
        width=1920,
        height=1080,
        fps=29.97,
        frame_count=300,
        duration_seconds=10.01,
        is_live=False,
    )

    assert video.fps > 0