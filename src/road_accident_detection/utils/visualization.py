"""
Visualization utilities for RoadAccidentAI.

This module provides reusable drawing utilities for visualizing computer
vision results. These utilities are intentionally detector-independent and
operate on the project's own domain models instead of Ultralytics-specific
objects.

The visualization layer is responsible only for rendering. It does not
perform detection, tracking, or accident reasoning.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from road_accident_detection.models.detection import Detection

__all__ = [
    "draw_detection",
    "draw_detections",
    "draw_text",
    "draw_fps",
]

###############################################################################
# Drawing Constants
###############################################################################

_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE: Final[float] = 0.5
_FONT_THICKNESS: Final[int] = 2

_DEFAULT_COLOR: Final[tuple[int, int, int]] = (0, 255, 0)
_TEXT_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)
_BACKGROUND_COLOR: Final[tuple[int, int, int]] = (0, 0, 0)

_LINE_THICKNESS: Final[int] = 2


def draw_detection(
    image: NDArray[np.uint8],
    detection: Detection,
    color: tuple[int, int, int] = _DEFAULT_COLOR,
) -> NDArray[np.uint8]:
    """
    Draw a single detection.

    Args:
        image:
            Input image.

        detection:
            Detection to draw.

        color:
            Bounding-box color in BGR.

    Returns:
        Image with the rendered detection.
    """

    x1, y1, x2, y2 = map(int, detection.bounding_box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        _LINE_THICKNESS,
    )

    label = (
        f"{detection.class_name} "
        f"{detection.confidence:.2f}"
    )

    if detection.is_tracked:
        label = (
            f"ID:{detection.tracker_id} "
            f"{label}"
        )

    draw_text(
        image=image,
        text=label,
        position=(x1, max(y1 - 10, 20)),
        background_color=color,
    )

    return image


def draw_detections(
    image: NDArray[np.uint8],
    detections: list[Detection],
    color: tuple[int, int, int] = _DEFAULT_COLOR,
) -> NDArray[np.uint8]:
    """
    Draw multiple detections.

    Args:
        image:
            Input image.

        detections:
            Detection list.

        color:
            Bounding-box color.

    Returns:
        Annotated image.
    """

    for detection in detections:
        draw_detection(
            image=image,
            detection=detection,
            color=color,
        )

    return image


def draw_text(
    image: NDArray[np.uint8],
    text: str,
    position: tuple[int, int],
    *,
    text_color: tuple[int, int, int] = _TEXT_COLOR,
    background_color: tuple[int, int, int] = _BACKGROUND_COLOR,
) -> NDArray[np.uint8]:
    """
    Draw text with a filled background.

    Args:
        image:
            Input image.

        text:
            Text to display.

        position:
            Top-left text position.

        text_color:
            Font color.

        background_color:
            Background rectangle color.

    Returns:
        Annotated image.
    """

    (width, height), baseline = cv2.getTextSize(
        text,
        _FONT,
        _FONT_SCALE,
        _FONT_THICKNESS,
    )

    x, y = position

    cv2.rectangle(
        image,
        (x, y - height - baseline),
        (x + width + 6, y + baseline),
        background_color,
        thickness=-1,
    )

    cv2.putText(
        image,
        text,
        (x + 3, y),
        _FONT,
        _FONT_SCALE,
        text_color,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )

    return image


def draw_fps(
    image: NDArray[np.uint8],
    fps: float,
) -> NDArray[np.uint8]:
    """
    Draw the current FPS.

    Args:
        image:
            Input image.

        fps:
            Current frames per second.

    Returns:
        Annotated image.
    """

    return draw_text(
        image=image,
        text=f"FPS: {fps:.2f}",
        position=(10, 25),
    )