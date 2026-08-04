"""
Detector package for RoadAccidentAI.

This package provides the detector abstraction layer used throughout the
RoadAccidentAI project.

The detector package isolates the rest of the application from any
particular object detection framework. Although the initial implementation
uses the latest stable Ultralytics YOLO, future detector implementations can
be integrated without changing the pipeline or downstream research modules.

Responsibilities
----------------
- Abstract detector interface
- Detector result model
- Detector factory
- Ultralytics YOLO implementation

Design Principles
-----------------
- Framework independence
- Dependency inversion
- Strong typing
- Clean interfaces
- Unit-testable
- Easily extensible

Public API
----------
The package exports the detector interface, detector result container,
factory, and concrete detector implementations.

Future detector implementations may include:

- RT-DETR
- YOLOv12
- Detectron2
- Grounding DINO
- Custom research detectors

without affecting the remainder of the application.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from .base import *
from .factory import *
from .result import *
from .ultralytics_yolo import *

__all__: list[str] = []