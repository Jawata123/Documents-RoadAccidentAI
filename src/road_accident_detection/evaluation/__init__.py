"""
RoadAccidentAI Evaluation Package
=================================

This package provides evaluation utilities for benchmarking object
detection performance, computing quantitative metrics, generating
visualizations, and producing evaluation reports.

Modules
-------
benchmark
    Runtime performance benchmarking.

confusion_matrix
    Confusion matrix generation.

evaluator
    End-to-end evaluation pipeline.

inference
    Model inference utilities.

map_metrics
    Mean Average Precision (mAP) computation.

metrics
    Detection performance metrics.

performance
    Performance summary generation.

plots
    Visualization and plotting utilities.

report
    Evaluation report generation.

statistics
    Statistical analysis utilities.

timer
    Timing and latency measurements.

validator
    Prediction and annotation validation.

visualizer
    Detection visualization utilities.
"""

from .benchmark import Benchmark
from .confusion_matrix import ConfusionMatrixEvaluator
from .evaluator import Evaluator
from .inference import InferenceEngine
from .map_metrics import MeanAveragePrecision
from .metrics import DetectionMetrics
from .performance import PerformanceSummary
from .plots import PlotGenerator
from .report import EvaluationReport
from .statistics import StatisticsAnalyzer
from .timer import PerformanceTimer
from .validator import PredictionValidator
from .visualizer import DetectionVisualizer

__all__ = [
    "Benchmark",
    "ConfusionMatrixEvaluator",
    "DetectionMetrics",
    "DetectionVisualizer",
    "EvaluationReport",
    "Evaluator",
    "InferenceEngine",
    "MeanAveragePrecision",
    "PerformanceSummary",
    "PerformanceTimer",
    "PlotGenerator",
    "PredictionValidator",
    "StatisticsAnalyzer",
]

__version__ = "1.0.0"