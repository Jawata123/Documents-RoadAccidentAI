"""
Detection performance metrics for RoadAccidentAI.

This module computes classification metrics from a confusion matrix
including accuracy, precision, recall, specificity, F1-score,
balanced accuracy, Matthews Correlation Coefficient (MCC),
Cohen's Kappa, and Negative Predictive Value (NPV).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionMetrics:
    """
    Detection performance metrics.

    Parameters
    ----------
    tp
        True Positives.

    fp
        False Positives.

    tn
        True Negatives.

    fn
        False Negatives.
    """

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    @property
    def specificity(self) -> float:
        denominator = self.tn + self.fp
        if denominator == 0:
            return 0.0
        return self.tn / denominator

    @property
    def sensitivity(self) -> float:
        return self.recall

    @property
    def negative_predictive_value(self) -> float:
        denominator = self.tn + self.fn
        if denominator == 0:
            return 0.0
        return self.tn / denominator

    @property
    def false_positive_rate(self) -> float:
        denominator = self.fp + self.tn
        if denominator == 0:
            return 0.0
        return self.fp / denominator

    @property
    def false_negative_rate(self) -> float:
        denominator = self.fn + self.tp
        if denominator == 0:
            return 0.0
        return self.fn / denominator

    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall

        if (p + r) == 0:
            return 0.0

        return 2.0 * p * r / (p + r)

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2.0

    @property
    def matthews_correlation_coefficient(self) -> float:

        numerator = (
            self.tp * self.tn
            - self.fp * self.fn
        )

        denominator = math.sqrt(
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )

        if denominator == 0:
            return 0.0

        return numerator / denominator

    @property
    def cohen_kappa(self) -> float:

        total = self.total

        if total == 0:
            return 0.0

        observed = self.accuracy

        expected = (
            (
                (self.tp + self.fp)
                * (self.tp + self.fn)
            )
            + (
                (self.fn + self.tn)
                * (self.fp + self.tn)
            )
        ) / (total * total)

        if expected == 1:
            return 0.0

        return (observed - expected) / (
            1 - expected
        )

    def to_dict(self) -> dict[str, float]:
        """
        Return all metrics.
        """

        return {

            "accuracy": self.accuracy,

            "precision": self.precision,

            "recall": self.recall,

            "specificity": self.specificity,

            "sensitivity": self.sensitivity,

            "negative_predictive_value":
                self.negative_predictive_value,

            "false_positive_rate":
                self.false_positive_rate,

            "false_negative_rate":
                self.false_negative_rate,

            "f1_score": self.f1_score,

            "balanced_accuracy":
                self.balanced_accuracy,

            "matthews_correlation_coefficient":
                self.matthews_correlation_coefficient,

            "cohen_kappa":
                self.cohen_kappa,
        }

    def print_summary(self) -> None:
        """
        Log evaluation metrics.
        """

        logger.info(
            "========== Detection Metrics =========="
        )

        for name, value in self.to_dict().items():

            logger.info(
                "%-35s : %.4f",
                name,
                value,
            )

        logger.info(
            "======================================="
        )