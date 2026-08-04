"""
Data models package for RoadAccidentAI.

This package contains the project's domain models. These models represent
structured data exchanged between different layers of the application,
including the video pipeline, vision system, tracking modules, event
reasoning, and evaluation components.

The models are intentionally independent of implementation details such as
Ultralytics YOLO, OpenCV, or any future tracking algorithm. This allows
internal implementations to evolve without affecting the rest of the
application.

Responsibilities
----------------
- Video metadata
- Frame metadata
- Detection objects
- Vehicle objects

Future research modules may introduce additional domain models such as:

- Track
- Trajectory
- SpeedEstimate
- AccelerationEstimate
- AccidentEvent
- EventConfidence
- Alert
- EvaluationResult
- BenchmarkResult

Design Principles
-----------------
- Immutable where appropriate
- Strong typing
- Lightweight
- Serialization-friendly
- Framework independent
- Unit-testable

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from .detection import *
from .frame import *
from .vehicle import *
from .video import *

__all__: list[str] = []