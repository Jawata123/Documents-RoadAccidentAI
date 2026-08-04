"""
Vision package for RoadAccidentAI.

This package contains the computer vision components of the application.
Its primary responsibility is performing inference and converting detector-
specific outputs into the project's standardized domain models.

The vision package is intentionally designed around abstraction rather than
a specific object detection framework. Although the initial implementation
uses the latest stable Ultralytics YOLO, the architecture allows future
replacement with other detectors without affecting the rest of the system.

Responsibilities
----------------
- Detector abstraction
- Detector implementations
- Detection result conversion
- Detector factory

The vision package should NOT contain:

- Multi-object tracking
- Speed estimation
- Trajectory analysis
- IoU computation
- Temporal reasoning
- Accident decision logic
- Alert generation

Those responsibilities belong to future research modules.

Design Principles
-----------------
- Detector independence
- Strong typing
- Modular architecture
- Framework abstraction
- Unit-testable
- Future extensibility

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from .detector import *

__all__: list[str] = []