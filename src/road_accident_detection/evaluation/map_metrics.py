"""
Mean Average Precision (mAP) evaluation module.

This module computes Average Precision (AP), mAP@0.5 and
mAP@0.5:0.95 for object detection evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BoundingBox:
    """
    Represents one object bounding box.
    """

    class_id: int

    confidence: float

    x1: float

    y1: float

    x2: float

    y2: float


class MeanAveragePrecision:
    """
    Mean Average Precision evaluator.
    """

    def __init__(
        self,
        iou_threshold: float = 0.50,
    ) -> None:

        self.iou_threshold = iou_threshold

        logger.info(
            "mAP evaluator initialized (IoU=%.2f).",
            iou_threshold,
        )

    @staticmethod
    def compute_iou(
        prediction: BoundingBox,
        target: BoundingBox,
    ) -> float:
        """
        Compute Intersection over Union.
        """

        x_left = max(prediction.x1, target.x1)
        y_top = max(prediction.y1, target.y1)

        x_right = min(prediction.x2, target.x2)
        y_bottom = min(prediction.y2, target.y2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (
            (x_right - x_left)
            * (y_bottom - y_top)
        )

        prediction_area = (
            (prediction.x2 - prediction.x1)
            * (prediction.y2 - prediction.y1)
        )

        target_area = (
            (target.x2 - target.x1)
            * (target.y2 - target.y1)
        )

        union = (
            prediction_area
            + target_area
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    def average_precision(
        self,
        predictions: list[BoundingBox],
        targets: list[BoundingBox],
    ) -> float:
        """
        Compute Average Precision.
        """

        if not predictions:
            return 0.0

        predictions = sorted(
            predictions,
            key=lambda x: x.confidence,
            reverse=True,
        )

        matched_targets: set[int] = set()

        true_positive = []

        false_positive = []

        for prediction in predictions:

            best_iou = 0.0

            best_index = -1

            for index, target in enumerate(targets):

                if index in matched_targets:
                    continue

                if prediction.class_id != target.class_id:
                    continue

                iou = self.compute_iou(
                    prediction,
                    target,
                )

                if iou > best_iou:

                    best_iou = iou

                    best_index = index

            if (
                best_iou >= self.iou_threshold
                and best_index >= 0
            ):

                matched_targets.add(best_index)

                true_positive.append(1)

                false_positive.append(0)

            else:

                true_positive.append(0)

                false_positive.append(1)

        tp = np.cumsum(true_positive)

        fp = np.cumsum(false_positive)

        if len(targets) == 0:
            return 0.0

        recalls = tp / len(targets)

        precisions = tp / (tp + fp + 1e-12)

        recalls = np.concatenate(
            ([0.0], recalls, [1.0])
        )

        precisions = np.concatenate(
            ([1.0], precisions, [0.0])
        )

        for index in range(
            len(precisions) - 2,
            -1,
            -1,
        ):

            precisions[index] = max(
                precisions[index],
                precisions[index + 1],
            )

        indices = np.where(
            recalls[1:] != recalls[:-1]
        )[0]

        ap = np.sum(
            (
                recalls[indices + 1]
                - recalls[indices]
            )
            * precisions[indices + 1]
        )

        return float(ap)

    def mean_average_precision(
        self,
        predictions: list[BoundingBox],
        targets: list[BoundingBox],
    ) -> float:
        """
        Compute mAP for current IoU threshold.
        """

        return self.average_precision(
            predictions,
            targets,
        )

    def map_50_to_95(
        self,
        predictions: list[BoundingBox],
        targets: list[BoundingBox],
    ) -> float:
        """
        Compute COCO-style mAP@0.5:0.95.
        """

        scores = []

        original_threshold = self.iou_threshold

        for threshold in np.arange(
            0.50,
            1.00,
            0.05,
        ):

            self.iou_threshold = float(
                threshold
            )

            scores.append(
                self.average_precision(
                    predictions,
                    targets,
                )
            )

        self.iou_threshold = original_threshold

        return float(np.mean(scores))

    def summary(
        self,
        predictions: list[BoundingBox],
        targets: list[BoundingBox],
    ) -> dict[str, float]:
        """
        Generate evaluation summary.
        """

        map50 = self.mean_average_precision(
            predictions,
            targets,
        )

        map5095 = self.map_50_to_95(
            predictions,
            targets,
        )

        logger.info(
            "mAP@0.5 = %.4f | mAP@0.5:0.95 = %.4f",
            map50,
            map5095,
        )

        return {
            "mAP@0.5": map50,
            "mAP@0.5:0.95": map5095,
        }