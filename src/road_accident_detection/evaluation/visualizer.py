"""
Detection visualization utilities for RoadAccidentAI.

This module generates annotated images showing object detections,
bounding boxes, confidence scores, and class labels. It can also save
successful predictions and failure cases for qualitative evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from road_accident_detection.evaluation.map_metrics import BoundingBox

logger = logging.getLogger(__name__)


class DetectionVisualizer:
    """
    Visualize object detection results.
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
            "Visualization output directory: %s",
            self.output_directory,
        )

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: Iterable[BoundingBox],
        class_names: dict[int, str] | None = None,
    ) -> np.ndarray:
        """
        Draw detections on an image.

        Parameters
        ----------
        image
            Input image.

        detections
            Iterable of BoundingBox objects.

        class_names
            Optional class ID to name mapping.

        Returns
        -------
        np.ndarray
            Annotated image.
        """

        output = image.copy()

        for detection in detections:

            x1 = int(detection.x1)
            y1 = int(detection.y1)
            x2 = int(detection.x2)
            y2 = int(detection.y2)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            class_name = (
                class_names.get(
                    detection.class_id,
                    str(detection.class_id),
                )
                if class_names
                else str(detection.class_id)
            )

            label = (
                f"{class_name} "
                f"{detection.confidence:.2f}"
            )

            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return output

    def save_image(
        self,
        image: np.ndarray,
        filename: str,
    ) -> Path:
        """
        Save an image.
        """

        output_path = (
            self.output_directory / filename
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

        logger.info(
            "Saved visualization: %s",
            output_path,
        )

        return output_path

    def save_detection_result(
        self,
        image: np.ndarray,
        detections: Iterable[BoundingBox],
        filename: str,
        class_names: dict[int, str] | None = None,
    ) -> Path:
        """
        Draw detections and save the image.
        """

        annotated = self.draw_detections(
            image=image,
            detections=detections,
            class_names=class_names,
        )

        return self.save_image(
            annotated,
            filename,
        )

    def save_success_case(
        self,
        image: np.ndarray,
        detections: Iterable[BoundingBox],
        index: int,
        class_names: dict[int, str] | None = None,
    ) -> Path:
        """
        Save a successful prediction.
        """

        filename = (
            f"success_{index:05d}.jpg"
        )

        return self.save_detection_result(
            image=image,
            detections=detections,
            filename=filename,
            class_names=class_names,
        )

    def save_failure_case(
        self,
        image: np.ndarray,
        detections: Iterable[BoundingBox],
        index: int,
        class_names: dict[int, str] | None = None,
    ) -> Path:
        """
        Save an incorrect prediction.
        """

        filename = (
            f"failure_{index:05d}.jpg"
        )

        return self.save_detection_result(
            image=image,
            detections=detections,
            filename=filename,
            class_names=class_names,
        )

    def save_batch(
        self,
        images: list[np.ndarray],
        detections: list[list[BoundingBox]],
        class_names: dict[int, str] | None = None,
    ) -> list[Path]:
        """
        Save multiple annotated images.
        """

        if len(images) != len(detections):
            raise ValueError(
                "Images and detections must have the same length."
            )

        output_paths: list[Path] = []

        for index, (image, boxes) in enumerate(
            zip(images, detections)
        ):

            filename = (
                f"prediction_{index:05d}.jpg"
            )

            output_paths.append(
                self.save_detection_result(
                    image=image,
                    detections=boxes,
                    filename=filename,
                    class_names=class_names,
                )
            )

        logger.info(
            "Saved %d annotated images.",
            len(output_paths),
        )

        return output_paths