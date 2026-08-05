"""
Benchmarking utilities for RoadAccidentAI.

This module measures runtime performance of the object detection
pipeline including preprocessing, inference, postprocessing,
latency, throughput (FPS), and overall execution statistics.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkResult:
    """
    Stores benchmarking statistics.
    """

    total_frames: int = 0

    preprocess_times: list[float] = field(default_factory=list)

    inference_times: list[float] = field(default_factory=list)

    postprocess_times: list[float] = field(default_factory=list)

    total_times: list[float] = field(default_factory=list)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        return statistics.stdev(values)

    @property
    def average_preprocess_time(self) -> float:
        return self._mean(self.preprocess_times)

    @property
    def average_inference_time(self) -> float:
        return self._mean(self.inference_times)

    @property
    def average_postprocess_time(self) -> float:
        return self._mean(self.postprocess_times)

    @property
    def average_total_time(self) -> float:
        return self._mean(self.total_times)

    @property
    def fps(self) -> float:
        if self.average_total_time <= 0:
            return 0.0
        return 1.0 / self.average_total_time

    @property
    def inference_std(self) -> float:
        return self._std(self.inference_times)

    @property
    def latency_ms(self) -> float:
        return self.average_total_time * 1000.0

    def to_dict(self) -> dict[str, float]:
        """
        Convert benchmark statistics into dictionary.
        """

        return {
            "frames": self.total_frames,
            "average_preprocess_time": self.average_preprocess_time,
            "average_inference_time": self.average_inference_time,
            "average_postprocess_time": self.average_postprocess_time,
            "average_total_time": self.average_total_time,
            "latency_ms": self.latency_ms,
            "fps": self.fps,
            "inference_std": self.inference_std,
        }


class Benchmark:
    """
    Runtime benchmark manager.

    Example
    -------
    >>> benchmark = Benchmark()
    >>> benchmark.start()
    >>> ...
    >>> benchmark.stop_inference()
    >>> results = benchmark.results
    """

    def __init__(self) -> None:

        self.results = BenchmarkResult()

        self._preprocess_start = 0.0
        self._inference_start = 0.0
        self._postprocess_start = 0.0
        self._total_start = 0.0

    def start(self) -> None:
        """
        Start benchmarking for a frame.
        """

        self._total_start = time.perf_counter()

    def start_preprocess(self) -> None:

        self._preprocess_start = time.perf_counter()

    def stop_preprocess(self) -> None:

        elapsed = time.perf_counter() - self._preprocess_start

        self.results.preprocess_times.append(elapsed)

    def start_inference(self) -> None:

        self._inference_start = time.perf_counter()

    def stop_inference(self) -> None:

        elapsed = time.perf_counter() - self._inference_start

        self.results.inference_times.append(elapsed)

    def start_postprocess(self) -> None:

        self._postprocess_start = time.perf_counter()

    def stop_postprocess(self) -> None:

        elapsed = time.perf_counter() - self._postprocess_start

        self.results.postprocess_times.append(elapsed)

    def stop(self) -> None:
        """
        Finish benchmarking for current frame.
        """

        elapsed = time.perf_counter() - self._total_start

        self.results.total_times.append(elapsed)

        self.results.total_frames += 1

    def reset(self) -> None:
        """
        Reset benchmark statistics.
        """

        logger.info("Resetting benchmark statistics.")

        self.results = BenchmarkResult()

    def summary(self) -> dict[str, float]:
        """
        Return benchmark summary.
        """

        logger.info(
            "Benchmark complete: %.2f FPS | %.2f ms latency",
            self.results.fps,
            self.results.latency_ms,
        )

        return self.results.to_dict()