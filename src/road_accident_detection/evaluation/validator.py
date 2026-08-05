"""
Prediction validation module for RoadAccidentAI.

This module validates predictions and annotations before evaluation.
It ensures that bounding boxes, confidence scores, class IDs, and
prediction lengths are valid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from road_accident_detection.evaluation.map_metrics import BoundingBox

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationResult:
    """
    Validation summary.
    """

    valid: bool

    errors: list[str]

    warnings: list[str]

    total_predictions: int

    total_ground_truth: int

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class PredictionValidator:
    """
    Validates predictions and annotations.
    """

    def __init__(self) -> None:

        logger.info(
            "Prediction validator initialized."
        )

    @staticmethod
    def _validate_box(
        box: BoundingBox,
        prefix: str,
    ) -> list[str]:

        errors: list[str] = []

        if box.x1 >= box.x2:

            errors.append(
                f"{prefix}: x1 must be smaller than x2."
            )

        if box.y1 >= box.y2:

            errors.append(
                f"{prefix}: y1 must be smaller than y2."
            )

        if not (0.0 <= box.confidence <= 1.0):

            errors.append(
                f"{prefix}: confidence must be between 0 and 1."
            )

        if box.class_id < 0:

            errors.append(
                f"{prefix}: invalid class id."
            )

        return errors

    def validate(
        self,
        predictions: list[BoundingBox],
        ground_truth: list[BoundingBox],
    ) -> ValidationResult:
        """
        Validate predictions and annotations.
        """

        errors: list[str] = []

        warnings: list[str] = []

        for index, box in enumerate(predictions):

            errors.extend(
                self._validate_box(
                    box,
                    f"Prediction {index}",
                )
            )

        for index, box in enumerate(ground_truth):

            errors.extend(
                self._validate_box(
                    box,
                    f"GroundTruth {index}",
                )
            )

        if len(predictions) == 0:

            warnings.append(
                "No predictions available."
            )

        if len(ground_truth) == 0:

            warnings.append(
                "No ground truth annotations available."
            )

        valid = len(errors) == 0

        logger.info(
            "Validation completed "
            "(valid=%s, errors=%d, warnings=%d).",
            valid,
            len(errors),
            len(warnings),
        )

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            total_predictions=len(predictions),
            total_ground_truth=len(ground_truth),
        )

    @staticmethod
    def print_summary(
        result: ValidationResult,
    ) -> None:
        """
        Print validation summary.
        """

        logger.info(
            "========== Validation =========="
        )

        logger.info(
            "Valid               : %s",
            result.valid,
        )

        logger.info(
            "Predictions         : %d",
            result.total_predictions,
        )

        logger.info(
            "Ground Truth        : %d",
            result.total_ground_truth,
        )

        logger.info(
            "Errors              : %d",
            len(result.errors),
        )

        logger.info(
            "Warnings            : %d",
            len(result.warnings),
        )

        if result.errors:

            logger.info("")

            logger.info("Errors:")

            for error in result.errors:

                logger.info(
                    "  • %s",
                    error,
                )

        if result.warnings:

            logger.info("")

            logger.info("Warnings:")

            for warning in result.warnings:

                logger.info(
                    "  • %s",
                    warning,
                )

        logger.info(
            "================================"
        )