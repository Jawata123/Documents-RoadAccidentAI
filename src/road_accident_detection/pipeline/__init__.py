"""
Pipeline package for RoadAccidentAI.

This package provides the application's processing pipeline responsible for
coordinating data flow between video sources, detectors, and future research
modules.

The pipeline orchestrates the processing of frames while remaining
independent of any specific object detection framework. It communicates with
detectors only through the detector abstraction layer defined in the
``vision.detector`` package.

Responsibilities
----------------
- Video source management
- Frame acquisition
- Processing orchestration
- Detector invocation
- Pipeline execution

The pipeline intentionally does NOT contain:

- Object detection algorithms
- Multi-object tracking
- Speed estimation
- Trajectory analysis
- IoU analysis
- Temporal reasoning
- Accident decision logic
- Alert generation
- Visualization logic

Those responsibilities belong to their respective packages.

Design Principles
-----------------
- Separation of concerns
- Dependency inversion
- Extensible processing stages
- Framework independence
- Unit-testable
- Research-ready architecture

Future pipeline extensions may include:

- Multi-camera synchronization
- Parallel processing
- Frame buffering
- Pipeline middleware
- Event dispatching
- Performance monitoring
- Batch processing

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from .frame import *
from .runner import *
from .source import *

__all__: list[str] = []