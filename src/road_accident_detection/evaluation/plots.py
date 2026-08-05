"""
Plot generation utilities for RoadAccidentAI.

This module creates publication-quality visualizations for evaluation
results including confusion matrices, precision-recall curves,
performance comparisons, latency analysis, FPS charts,
confidence distributions, and mAP summaries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class PlotGenerator:
    """
    Generates evaluation plots.
    """

    def __init__(
        self,
        output_directory: str | Path,
    ) -> None:

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Plot output directory: %s",
            self.output_directory,
        )

    @staticmethod
    def _save(
        figure: plt.Figure,
        output_file: Path,
    ) -> None:

        figure.tight_layout()

        figure.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)

        logger.info(
            "Saved %s",
            output_file.name,
        )

    def response_time_chart(
        self,
        response_times: list[float],
    ) -> Path:

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        axis.plot(
            response_times,
            linewidth=2,
        )

        axis.set_title(
            "Response Time Per Frame"
        )

        axis.set_xlabel("Frame")

        axis.set_ylabel(
            "Response Time (ms)"
        )

        axis.grid(True)

        output = (
            self.output_directory
            / "response_time.png"
        )

        self._save(
            figure,
            output,
        )

        return output

    def fps_chart(
        self,
        fps_values: list[float],
    ) -> Path:

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        axis.plot(
            fps_values,
            linewidth=2,
        )

        axis.set_title(
            "Frames Per Second"
        )

        axis.set_xlabel("Measurement")

        axis.set_ylabel("FPS")

        axis.grid(True)

        output = (
            self.output_directory
            / "fps_chart.png"
        )

        self._save(
            figure,
            output,
        )

        return output

    def confidence_distribution(
        self,
        confidence_scores: list[float],
    ) -> Path:

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        axis.hist(
            confidence_scores,
            bins=20,
        )

        axis.set_title(
            "Detection Confidence Distribution"
        )

        axis.set_xlabel(
            "Confidence"
        )

        axis.set_ylabel(
            "Frequency"
        )

        output = (
            self.output_directory
            / "confidence_distribution.png"
        )

        self._save(
            figure,
            output,
        )

        return output

    def precision_recall_curve(
        self,
        recall: list[float],
        precision: list[float],
    ) -> Path:

        figure, axis = plt.subplots(
            figsize=(6, 6)
        )

        axis.plot(
            recall,
            precision,
            linewidth=2,
        )

        axis.set_title(
            "Precision-Recall Curve"
        )

        axis.set_xlabel(
            "Recall"
        )

        axis.set_ylabel(
            "Precision"
        )

        axis.set_xlim(
            0,
            1,
        )

        axis.set_ylim(
            0,
            1,
        )

        axis.grid(True)

        output = (
            self.output_directory
            / "precision_recall_curve.png"
        )

        self._save(
            figure,
            output,
        )

        return output

    def map_chart(
        self,
        map50: float,
        map5095: float,
    ) -> Path:

        figure, axis = plt.subplots(
            figsize=(6, 5)
        )

        axis.bar(
            [
                "mAP@0.50",
                "mAP@0.50:0.95",
            ],
            [
                map50,
                map5095,
            ],
        )

        axis.set_ylim(
            0,
            1,
        )

        axis.set_ylabel(
            "Score"
        )

        axis.set_title(
            "Mean Average Precision"
        )

        output = (
            self.output_directory
            / "map_chart.png"
        )

        self._save(
            figure,
            output,
        )

        return output

    def performance_bar_chart(
        self,
        metrics: dict[str, float],
    ) -> Path:

        names = list(metrics.keys())

        values = list(metrics.values())

        figure, axis = plt.subplots(
            figsize=(10, 5)
        )

        axis.bar(
            names,
            values,
        )

        axis.set_ylim(
            0,
            1,
        )

        axis.set_ylabel(
            "Score"
        )

        axis.set_title(
            "Performance Metrics"
        )

        plt.xticks(
            rotation=30,
            ha="right",
        )

        output = (
            self.output_directory
            / "performance_metrics.png"
        )

        self._save(
            figure,
            output,
        )

        return output

    def latency_distribution(
        self,
        latency: list[float],
    ) -> Path:

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        axis.boxplot(
            latency,
        )

        axis.set_ylabel(
            "Latency (ms)"
        )

        axis.set_title(
            "Latency Distribution"
        )

        output = (
            self.output_directory
            / "latency_distribution.png"
        )

        self._save(
            figure,
            output,
        )

        return output

    def detection_count_chart(
        self,
        detections: list[int],
    ) -> Path:

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        axis.plot(
            detections,
            linewidth=2,
        )

        axis.set_title(
            "Detected Objects Per Frame"
        )

        axis.set_xlabel(
            "Frame"
        )

        axis.set_ylabel(
            "Detections"
        )

        axis.grid(True)

        output = (
            self.output_directory
            / "detections_per_frame.png"
        )

        self._save(
            figure,
            output,
        )

        return output

    def benchmark_summary(
        self,
        preprocess: float,
        inference: float,
        postprocess: float,
    ) -> Path:

        figure, axis = plt.subplots(
            figsize=(6, 5)
        )

        axis.bar(
            [
                "Pre",
                "Inference",
                "Post",
            ],
            [
                preprocess,
                inference,
                postprocess,
            ],
        )

        axis.set_ylabel(
            "Milliseconds"
        )

        axis.set_title(
            "Pipeline Timing"
        )

        output = (
            self.output_directory
            / "pipeline_timing.png"
        )

        self._save(
            figure,
            output,
        )

        return output