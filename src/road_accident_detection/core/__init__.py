"""
Core package for RoadAccidentAI.

This package contains the foundational infrastructure used throughout the
project. Every other package depends on the modules defined here.

Responsibilities
----------------
- Global constants
- Custom exception hierarchy
- Logging configuration
- Path management
- Shared type aliases
- Common validation utilities

The core package is intentionally lightweight and free from computer vision,
deep learning, or application-specific logic. This separation ensures the
foundation remains stable as future research modules are integrated.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from .constants import *
from .exceptions import *
from .logging import *
from .paths import *
from .types import *
from .validation import *

__all__: list[str] = []