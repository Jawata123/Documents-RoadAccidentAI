"""
Confusion Matrix Evaluation Module.

This module computes binary confusion matrices together with
common classification statistics for RoadAccidentAI evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConfusionMatrixResult:
    """
    Stores confusion matrix statistics.
    """

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.true_positive, self.false_negative],
                [self.false_positive, self.true_negative],
            ],
            dtype=np.int64,
        )

    @property
    def accuracy(self) -> float:
        total = (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )
        return (
            (self.true_positive + self.true_negative) / total
            if total > 0
            else 0.0
        )

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator > 0 else 0.0

    @property
    def specificity(self) -> float:
        denominator = self.true_negative + self.false_positive
        return self.true_negative / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall

        if (p + r) == 0:
            return 0.0

        return 2.0 * p * r / (p + r)

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1_score": self.f1_score,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
        }


class ConfusionMatrixEvaluator:
    """
    Binary confusion matrix evaluator.
    """

    def __init__(self) -> None:
        logger.info("Confusion Matrix Evaluator initialized.")

    @staticmethod
    def evaluate(
        ground_truth: list[int],
        predictions: list[int],
    ) -> ConfusionMatrixResult:
        """
        Compute confusion matrix statistics.

        Parameters
        ----------
        ground_truth
            Ground truth binary labels.

        predictions
            Predicted binary labels.

        Returns
        -------
        ConfusionMatrixResult
        """

        if len(ground_truth) != len(predictions):
            raise ValueError(
                "Ground truth and prediction lengths must match."
            )

        tp = fp = tn = fn = 0

        for gt, pred in zip(ground_truth, predictions):

            if gt == 1 and pred == 1:
                tp += 1

            elif gt == 0 and pred == 1:
                fp += 1

            elif gt == 0 and pred == 0:
                tn += 1

            elif gt == 1 and pred == 0:
                fn += 1

        logger.info(
            "Confusion Matrix | TP=%d FP=%d TN=%d FN=%d",
            tp,
            fp,
            tn,
            fn,
        )

        return ConfusionMatrixResult(
            true_positive=tp,
            false_positive=fp,
            true_negative=tn,
            false_negative=fn,
        )

    @staticmethod
    def save_plot(
        result: ConfusionMatrixResult,
        output_path: str,
    ) -> None:
        """
        Save confusion matrix figure.
        """

        matrix = result.matrix

        figure, axis = plt.subplots(figsize=(6, 5))

        image = axis.imshow(matrix, cmap="Blues")

        plt.colorbar(image)

        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])

        axis.set_xticklabels(
            [
                "Predicted\nPositive",
                "Predicted\nNegative",
            ]
        )

        axis.set_yticklabels(
            [
                "Actual\nPositive",
                "Actual\nNegative",
            ]
        )

        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="black",
                )

        axis.set_title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(output_path, dpi=300)

        plt.close(figure)

        logger.info(
            "Confusion matrix saved to %s",
            output_path,
        )

    @staticmethod
    def print_summary(
        result: ConfusionMatrixResult,
    ) -> None:
        """
        Print formatted confusion matrix summary.
        """

        logger.info("========== Evaluation ==========")

        logger.info(
            "Accuracy     : %.4f",
            result.accuracy,
        )

        logger.info(
            "Precision    : %.4f",
            result.precision,
        )

        logger.info(
            "Recall       : %.4f",
            result.recall,
        )

        logger.info(
            "Specificity  : %.4f",
            result.specificity,
        )

        logger.info(
            "F1 Score     : %.4f",
            result.f1_score,
        )

        logger.info("================================")