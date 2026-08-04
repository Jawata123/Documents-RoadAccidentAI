"""
Configuration package for RoadAccidentAI.

This package provides centralized configuration management for the entire
application. All runtime settings should be loaded through this package
instead of being hard-coded throughout the codebase.

Responsibilities
----------------
- Default configuration values
- Configuration schemas
- Configuration loading
- Configuration validation
- Type-safe settings access

Design Principles
-----------------
- Configuration-driven architecture
- Immutable runtime settings
- Strong validation
- Single source of truth
- Extensible for future research modules

Future configuration categories may include:

- YOLO detector settings
- Video source settings
- Tracking parameters
- Speed estimation
- Trajectory analysis
- IoU thresholds
- Temporal reasoning
- Accident decision logic
- Alert generation
- Benchmark configuration
- Dashboard configuration
- REST API configuration
- Docker configuration

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from .defaults import *
from .loader import *
from .schemas import *
from .settings import *

__all__: list[str] = []