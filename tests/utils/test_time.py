"""
Unit tests for road_accident_detection.utils.time.

These tests verify the project's time utility functions used for measuring
execution time, calculating FPS, formatting durations, and obtaining
timestamps. The tests ensure deterministic behavior and correctness across
supported platforms.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import time

from road_accident_detection.utils.time import (
    ExecutionTimer,
    calculate_fps,
    current_timestamp,
    format_duration,
)


def test_current_timestamp_returns_string() -> None:
    """
    Verify current_timestamp returns a non-empty string.
    """

    timestamp = current_timestamp()

    assert isinstance(timestamp, str)
    assert len(timestamp) > 0


def test_format_duration_seconds() -> None:
    """
    Verify formatting of short durations.
    """

    assert format_duration(5.25) == "00:00:05.250"


def test_format_duration_minutes() -> None:
    """
    Verify formatting of minute durations.
    """

    assert format_duration(125.5) == "00:02:05.500"


def test_format_duration_hours() -> None:
    """
    Verify formatting of hour durations.
    """

    assert format_duration(3661.125) == "01:01:01.125"


def test_calculate_fps_positive() -> None:
    """
    Verify FPS calculation.
    """

    fps = calculate_fps(
        frame_count=300,
        elapsed_time=10.0,
    )

    assert fps == 30.0


def test_calculate_fps_zero_elapsed() -> None:
    """
    Verify zero elapsed time does not raise an error.
    """

    fps = calculate_fps(
        frame_count=100,
        elapsed_time=0.0,
    )

    assert fps == 0.0


def test_execution_timer_elapsed() -> None:
    """
    Verify timer measures elapsed time.
    """

    timer = ExecutionTimer()

    timer.start()

    time.sleep(0.02)

    timer.stop()

    assert timer.elapsed_time > 0.0


def test_execution_timer_context_manager() -> None:
    """
    Verify context manager support.
    """

    with ExecutionTimer() as timer:
        time.sleep(0.01)

    assert timer.elapsed_time > 0.0


def test_execution_timer_restart() -> None:
    """
    Verify timer can be restarted.
    """

    timer = ExecutionTimer()

    timer.start()
    time.sleep(0.01)
    timer.stop()

    first = timer.elapsed_time

    timer.start()
    time.sleep(0.01)
    timer.stop()

    second = timer.elapsed_time

    assert first > 0.0
    assert second > 0.0


def test_execution_timer_repr() -> None:
    """
    Verify string representation.
    """

    timer = ExecutionTimer()

    representation = repr(timer)

    assert isinstance(representation, str)
    assert "ExecutionTimer" in representation


def test_calculate_fps_fractional() -> None:
    """
    Verify fractional FPS calculation.
    """

    fps = calculate_fps(
        frame_count=75,
        elapsed_time=2.5,
    )

    assert fps == 30.0


def test_calculate_fps_large_values() -> None:
    """
    Verify FPS calculation for large inputs.
    """

    fps = calculate_fps(
        frame_count=18000,
        elapsed_time=600.0,
    )

    assert fps == 30.0