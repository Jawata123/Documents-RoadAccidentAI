"""
Utility package for RoadAccidentAI.

This package contains reusable utility functions that are shared across the
project. Utility modules provide generic helper functionality and should
remain independent of business logic, computer vision algorithms, and
research-specific implementations.

Responsibilities
----------------
- Image utilities
- Time utilities
- Visualization utilities

Design Principles
-----------------
- Generic and reusable
- Framework independent where practical
- No business logic
- Unit-testable
- Side-effect free whenever possible

Future utility modules may include:

- File utilities
- Geometry utilities
- Math utilities
- Serialization helpers
- Performance timers
- Color utilities
- Drawing helpers
- Statistics utilities

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from .image import *
from .time import *
from .visualization import *

__all__: list[str] = []