"""
Time utility functions for RoadAccidentAI.

This module provides reusable utilities for working with timestamps,
durations, frame timing, and execution time measurements. These utilities are
generic and can be used throughout the project without depending on OpenCV,
Ultralytics, or any research-specific module.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

__all__ = [
    "ExecutionTimer",
    "current_timestamp",
    "format_duration",
    "fps_from_elapsed",
]


_SECONDS_PER_MINUTE: Final[int] = 60
_SECONDS_PER_HOUR: Final[int] = 3600


def current_timestamp() -> datetime:
    """
    Return the current UTC timestamp.

    Returns:
        Timezone-aware UTC datetime.
    """
    return datetime.now(UTC)


def format_duration(seconds: float) -> str:
    """
    Format a duration into a human-readable string.

    Args:
        seconds:
            Duration in seconds.

    Returns:
        Formatted duration string.

    Examples:
        >>> format_duration(65)
        '00:01:05'

        >>> format_duration(3725.5)
        '01:02:05'
    """

    if seconds < 0:
        raise ValueError("Duration cannot be negative.")

    total_seconds = int(seconds)

    hours = total_seconds // _SECONDS_PER_HOUR
    minutes = (total_seconds % _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE
    remaining_seconds = total_seconds % _SECONDS_PER_MINUTE

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


def fps_from_elapsed(
    frame_count: int,
    elapsed_seconds: float,
) -> float:
    """
    Calculate frames per second.

    Args:
        frame_count:
            Number of processed frames.

        elapsed_seconds:
            Processing time in seconds.

    Returns:
        Frames per second.

    Notes:
        Returns 0.0 if the elapsed time is zero.
    """

    if elapsed_seconds <= 0.0:
        return 0.0

    return frame_count / elapsed_seconds


@dataclass(slots=True)
class ExecutionTimer(ContextDecorator):
    """
    High-resolution execution timer.

    The timer may be used either directly or as a context manager.

    Example:
        >>> with ExecutionTimer() as timer:
        ...     process_video()
        >>> print(timer.elapsed)

    Example:
        >>> timer = ExecutionTimer()
        >>> timer.start()
        >>> do_work()
        >>> timer.stop()
        >>> print(timer.elapsed)
    """

    _start_time: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _end_time: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def start(self) -> None:
        """
        Start the timer.
        """
        self._end_time = None
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop the timer.

        Returns:
            Elapsed time in seconds.

        Raises:
            RuntimeError:
                If the timer has not been started.
        """

        if self._start_time is None:
            raise RuntimeError(
                "ExecutionTimer has not been started."
            )

        self._end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """
        Return the elapsed time.

        Returns:
            Elapsed time in seconds.

        Raises:
            RuntimeError:
                If the timer has not been started.
        """

        if self._start_time is None:
            raise RuntimeError(
                "ExecutionTimer has not been started."
            )

        end = (
            self._end_time
            if self._end_time is not None
            else time.perf_counter()
        )

        return end - self._start_time

    def reset(self) -> None:
        """
        Reset the timer.
        """
        self._start_time = None
        self._end_time = None

    def __enter__(self) -> "ExecutionTimer":
        """
        Enter the runtime context.

        Returns:
            The timer instance.
        """
        self.start()
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
            False so any exception is propagated.
        """
        self.stop()
        return False