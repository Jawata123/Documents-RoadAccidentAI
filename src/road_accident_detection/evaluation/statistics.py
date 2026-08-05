"""
Statistical analysis utilities for RoadAccidentAI.

This module computes descriptive statistics used during model
evaluation and benchmarking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import sqrt
from statistics import mean, median, stdev
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StatisticsSummary:
    """
    Statistical summary.
    """

    count: int

    minimum: float

    maximum: float

    mean: float

    median: float

    standard_deviation: float

    variance: float

    percentile25: float

    percentile75: float

    confidence_interval_low: float

    confidence_interval_high: float

    def to_dict(self) -> dict[str, float]:

        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "standard_deviation": self.standard_deviation,
            "variance": self.variance,
            "percentile25": self.percentile25,
            "percentile75": self.percentile75,
            "confidence_interval_low":
                self.confidence_interval_low,
            "confidence_interval_high":
                self.confidence_interval_high,
        }


class StatisticsAnalyzer:
    """
    Computes descriptive statistics.
    """

    def __init__(self) -> None:

        logger.info(
            "Statistics analyzer initialized."
        )

    @staticmethod
    def percentile(
        values: Sequence[float],
        percentile: float,
    ) -> float:

        if not values:
            return 0.0

        sorted_values = sorted(values)

        k = (
            len(sorted_values) - 1
        ) * percentile

        lower = int(k)

        upper = min(
            lower + 1,
            len(sorted_values) - 1,
        )

        if lower == upper:
            return sorted_values[lower]

        fraction = k - lower

        return (
            sorted_values[lower]
            + (
                sorted_values[upper]
                - sorted_values[lower]
            )
            * fraction
        )

    @staticmethod
    def confidence_interval(
        values: Sequence[float],
        confidence: float = 0.95,
    ) -> tuple[float, float]:

        if len(values) < 2:
            return (0.0, 0.0)

        sample_mean = mean(values)

        sample_std = stdev(values)

        z = 1.96

        margin = (
            z
            * sample_std
            / sqrt(len(values))
        )

        return (
            sample_mean - margin,
            sample_mean + margin,
        )

    def summarize(
        self,
        values: Sequence[float],
    ) -> StatisticsSummary:

        if not values:

            raise ValueError(
                "Input sequence is empty."
            )

        ci_low, ci_high = (
            self.confidence_interval(values)
        )

        std = (
            stdev(values)
            if len(values) > 1
            else 0.0
        )

        variance = std ** 2

        summary = StatisticsSummary(

            count=len(values),

            minimum=min(values),

            maximum=max(values),

            mean=mean(values),

            median=median(values),

            standard_deviation=std,

            variance=variance,

            percentile25=self.percentile(
                values,
                0.25,
            ),

            percentile75=self.percentile(
                values,
                0.75,
            ),

            confidence_interval_low=ci_low,

            confidence_interval_high=ci_high,
        )

        logger.info(
            "Statistics computed for %d values.",
            summary.count,
        )

        return summary

    def print_summary(
        self,
        summary: StatisticsSummary,
    ) -> None:

        logger.info(
            "========== Statistics =========="
        )

        for key, value in (
            summary.to_dict().items()
        ):

            logger.info(
                "%-30s : %.4f",
                key,
                value,
            )

        logger.info(
            "================================"
        )