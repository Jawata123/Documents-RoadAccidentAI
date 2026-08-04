"""
Common type aliases used throughout the RoadAccidentAI project.

This module centralizes reusable type aliases to improve readability,
maintainability, and consistency across the codebase.

The aliases defined here are intentionally generic and independent of any
specific computer vision, tracking, or accident detection logic.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

__all__ = [
    "PathLike",
    "JSONDict",
    "JSONList",
    "JSONValue",
    "Numeric",
    "BoundingBox",
    "Point2D",
    "Size2D",
    "FrameIndex",
    "ObjectID",
    "Confidence",
]

###############################################################################
# Generic Types
###############################################################################

PathLike: TypeAlias = str | Path

Numeric: TypeAlias = int | float

###############################################################################
# JSON Types
###############################################################################

JSONValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | dict[str, Any]
    | list[Any]
)

JSONDict: TypeAlias = dict[str, JSONValue]

JSONList: TypeAlias = list[JSONValue]

###############################################################################
# Computer Vision Types
###############################################################################

BoundingBox: TypeAlias = tuple[
    float,
    float,
    float,
    float,
]
"""
Bounding box represented as:

(x_min, y_min, x_max, y_max)
"""

Point2D: TypeAlias = tuple[
    float,
    float,
]
"""
2D point represented as:

(x, y)
"""

Size2D: TypeAlias = tuple[
    int,
    int,
]
"""
Image size represented as:

(width, height)
"""

###############################################################################
# Domain Types
###############################################################################

FrameIndex: TypeAlias = int
"""
Video frame index.
"""

ObjectID: TypeAlias = int
"""
Unique tracked object identifier.
"""

Confidence: TypeAlias = float
"""
Detection confidence score.
"""