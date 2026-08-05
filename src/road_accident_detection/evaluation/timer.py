"""
High-resolution performance timer for RoadAccidentAI.

This module provides reusable timing utilities for measuring
preprocessing, inference, postprocessing, and overall execution time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TimerStatistics:
    """
    Stores timing statistics.
    """

    measurements: list[float] = field(default_factory=list)

    def add(self, elapsed: float) -> None:
        """
        Add one timing measurement.
        """
        self.measurements.append(elapsed)

    @property
    def count(self) -> int:
        return len(self.measurements)

    @property
    def total(self) -> float:
        return sum(self.measurements)

    @property
    def average(self) -> float:
        if not self.measurements:
            return 0.0
        return self.total / self.count

    @property
    def minimum(self) -> float:
        if not self.measurements:
            return 0.0
        return min(self.measurements)

    @property
    def maximum(self) -> float:
        if not self.measurements:
            return 0.0
        return max(self.measurements)

    @property
    def average_ms(self) -> float:
        return self.average * 1000.0

    @property
    def minimum_ms(self) -> float:
        return self.minimum * 1000.0

    @property
    def maximum_ms(self) -> float:
        return self.maximum * 1000.0

    @property
    def fps(self) -> float:
        if self.average <= 0:
            return 0.0
        return 1.0 / self.average

    def reset(self) -> None:
        """
        Clear all measurements.
        """
        self.measurements.clear()

    def to_dict(self) -> dict[str, float]:
        """
        Convert statistics to dictionary.
        """

        return {
            "count": self.count,
            "total_seconds": self.total,
            "average_seconds": self.average,
            "minimum_seconds": self.minimum,
            "maximum_seconds": self.maximum,
            "average_ms": self.average_ms,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "fps": self.fps,
        }


class PerformanceTimer:
    """
    High-resolution reusable timer.
    """

    def __init__(self) -> None:

        self._start_time = 0.0

        self.statistics = TimerStatistics()

    def start(self) -> None:
        """
        Start timing.
        """

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop timing and record elapsed time.

        Returns
        -------
        float
            Elapsed time in seconds.
        """

        elapsed = (
            time.perf_counter()
            - self._start_time
        )

        self.statistics.add(elapsed)

        return elapsed

    def measure(self, function, *args, **kwargs):
        """
        Measure execution time of a callable.

        Parameters
        ----------
        function
            Callable to execute.

        Returns
        -------
        Any
            Function return value.
        """

        self.start()

        result = function(
            *args,
            **kwargs,
        )

        elapsed = self.stop()

        logger.debug(
            "%s executed in %.6f seconds.",
            function.__name__,
            elapsed,
        )

        return result

    def reset(self) -> None:
        """
        Reset timer statistics.
        """

        self.statistics.reset()

    def summary(self) -> dict[str, float]:
        """
        Return timing summary.
        """

        summary = self.statistics.to_dict()

        logger.info(
            "Average %.3f ms | FPS %.2f",
            summary["average_ms"],
            summary["fps"],
        )

        return summary