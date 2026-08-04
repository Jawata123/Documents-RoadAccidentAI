"""
RoadAccidentAI
==============

A production-quality, extensible framework for real-time road accident
detection using the latest stable Ultralytics YOLO.

The package is designed to provide a modular architecture that supports
future research extensions without requiring major refactoring.

Current capabilities include:

- Configuration management
- Core infrastructure
- Video processing pipeline
- Vision abstraction
- Ultralytics YOLO integration
- Common data models
- Utility functions

Future research modules may include:

- Multi-object tracking
- Speed estimation
- Acceleration estimation
- Trajectory analysis
- IoU analysis
- Temporal reasoning
- Accident confidence scoring
- Event verification
- Alert generation
- Dataset benchmarking
- Dashboard integration
- REST API
- Docker deployment

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

__title__ = "RoadAccidentAI"
__package_name__ = "road_accident_detection"
__version__ = "0.1.0"
__author__ = "RoadAccidentAI Research Project"
__license__ = "MIT"
__status__ = "Development"

__all__: list[str] = [
    "__title__",
    "__package_name__",
    "__version__",
    "__author__",
    "__license__",
    "__status__",
]