"""
Performance summary module for RoadAccidentAI.

This module aggregates runtime benchmarking, classification metrics,
and object detection metrics into a unified performance report.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from road_accident_detection.evaluation.benchmark import BenchmarkResult
from road_accident_detection.evaluation.metrics import DetectionMetrics

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PerformanceSummary:
    """
    Aggregated evaluation summary.

    Parameters
    ----------
    benchmark
        Runtime benchmark results.

    metrics
        Detection metrics.

    map50
        Mean Average Precision @ IoU=0.50.

    map5095
        COCO-style Mean Average Precision @ IoU=0.50:0.95.
    """

    benchmark: BenchmarkResult

    metrics: DetectionMetrics

    map50: float

    map5095: float

    @property
    def average_latency_ms(self) -> float:
        return self.benchmark.latency_ms

    @property
    def average_fps(self) -> float:
        return self.benchmark.fps

    @property
    def average_inference_time_ms(self) -> float:
        return (
            self.benchmark.average_inference_time
            * 1000.0
        )

    @property
    def average_preprocess_time_ms(self) -> float:
        return (
            self.benchmark.average_preprocess_time
            * 1000.0
        )

    @property
    def average_postprocess_time_ms(self) -> float:
        return (
            self.benchmark.average_postprocess_time
            * 1000.0
        )

    def to_dict(self) -> dict[str, float]:
        """
        Return complete performance summary.
        """

        summary = {

            "frames_processed":
                self.benchmark.total_frames,

            "accuracy":
                self.metrics.accuracy,

            "precision":
                self.metrics.precision,

            "recall":
                self.metrics.recall,

            "specificity":
                self.metrics.specificity,

            "sensitivity":
                self.metrics.sensitivity,

            "f1_score":
                self.metrics.f1_score,

            "balanced_accuracy":
                self.metrics.balanced_accuracy,

            "mcc":
                self.metrics
                .matthews_correlation_coefficient,

            "cohen_kappa":
                self.metrics.cohen_kappa,

            "negative_predictive_value":
                self.metrics
                .negative_predictive_value,

            "false_positive_rate":
                self.metrics.false_positive_rate,

            "false_negative_rate":
                self.metrics.false_negative_rate,

            "map_50":
                self.map50,

            "map_50_95":
                self.map5095,

            "fps":
                self.average_fps,

            "latency_ms":
                self.average_latency_ms,

            "preprocess_time_ms":
                self.average_preprocess_time_ms,

            "inference_time_ms":
                self.average_inference_time_ms,

            "postprocess_time_ms":
                self.average_postprocess_time_ms,

        }

        return summary

    def print_summary(self) -> None:
        """
        Print formatted evaluation summary.
        """

        logger.info(
            "========== PERFORMANCE SUMMARY =========="
        )

        for key, value in self.to_dict().items():

            if isinstance(value, float):

                logger.info(
                    "%-30s : %.4f",
                    key,
                    value,
                )

            else:

                logger.info(
                    "%-30s : %s",
                    key,
                    value,
                )

        logger.info(
            "========================================="
        )

    def markdown_table(self) -> str:
        """
        Export performance summary as Markdown table.
        """

        lines = [
            "| Metric | Value |",
            "|-------|------:|",
        ]

        for key, value in self.to_dict().items():

            if isinstance(value, float):

                value_str = f"{value:.4f}"

            else:

                value_str = str(value)

            metric = (
                key.replace("_", " ")
                .title()
            )

            lines.append(
                f"| {metric} | {value_str} |"
            )

        return "\n".join(lines)

    def latex_table(self) -> str:
        """
        Export performance summary as a LaTeX table.
        """

        rows = []

        for key, value in self.to_dict().items():

            metric = (
                key.replace("_", " ")
                .title()
            )

            if isinstance(value, float):

                value_str = f"{value:.4f}"

            else:

                value_str = str(value)

            rows.append(
                f"{metric} & {value_str} \\\\"
            )

        body = "\n".join(rows)

        return (
            "\\begin{tabular}{lr}\n"
            "\\toprule\n"
            "Metric & Value\\\\\n"
            "\\midrule\n"
            f"{body}\n"
            "\\bottomrule\n"
            "\\end{tabular}"
        )