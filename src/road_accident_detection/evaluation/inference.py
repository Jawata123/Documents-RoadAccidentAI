"""
Inference engine for RoadAccidentAI.

This module provides a lightweight wrapper around the detector
implementation. It performs inference on image frames while measuring
runtime performance and returning standardized detection results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from road_accident_detection.evaluation.benchmark import Benchmark
from road_accident_detection.vision.detector.base import BaseDetector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceResult:
    """
    Result returned from a single inference.
    """

    frame: np.ndarray

    detections: list[Any]

    inference_time: float

    frame_index: int


class InferenceEngine:
    """
    Executes object detection inference.

    Parameters
    ----------
    detector
        Initialized detector instance.
    """

    def __init__(
        self,
        detector: BaseDetector,
    ) -> None:

        self._detector = detector

        self._benchmark = Benchmark()

        logger.info(
            "Inference engine initialized."
        )

    @property
    def benchmark(self) -> Benchmark:
        """
        Return benchmark instance.
        """

        return self._benchmark

    def predict(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> InferenceResult:
        """
        Run inference on one frame.

        Parameters
        ----------
        frame
            Input BGR frame.

        frame_index
            Frame number.

        Returns
        -------
        InferenceResult
        """

        self._benchmark.start()

        self._benchmark.start_preprocess()

        processed_frame = self._preprocess(frame)

        self._benchmark.stop_preprocess()

        self._benchmark.start_inference()

        detections = self._detector.detect(
            processed_frame
        )

        self._benchmark.stop_inference()

        self._benchmark.start_postprocess()

        detections = self._postprocess(
            detections
        )

        self._benchmark.stop_postprocess()

        self._benchmark.stop()

        inference_time = (
            self._benchmark.results.average_inference_time
        )

        logger.debug(
            "Frame %d processed with %d detections.",
            frame_index,
            len(detections),
        )

        return InferenceResult(
            frame=processed_frame,
            detections=detections,
            inference_time=inference_time,
            frame_index=frame_index,
        )

    def predict_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[InferenceResult]:
        """
        Run inference on multiple frames.
        """

        results: list[
            InferenceResult
        ] = []

        for index, frame in enumerate(frames):

            results.append(
                self.predict(
                    frame,
                    frame_index=index,
                )
            )

        return results

    @staticmethod
    def _preprocess(
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Image preprocessing.

        Future preprocessing steps may include:

        - Resize
        - Letterbox
        - Normalization
        - Color conversion
        """

        return frame

    @staticmethod
    def _postprocess(
        detections: list[Any],
    ) -> list[Any]:
        """
        Postprocess detector output.

        Future processing may include:

        - Confidence filtering
        - NMS
        - Sorting
        - Class filtering
        """

        return detections

    def reset_benchmark(
        self,
    ) -> None:
        """
        Reset runtime statistics.
        """

        self._benchmark.reset()

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Return benchmark summary.
        """

        return self._benchmark.summary()