"""
Utility test package for RoadAccidentAI.

This package contains unit tests for the project's utility modules.

The utility layer provides reusable helper functions that support image
processing, time measurement, visualization, and other common operations.
These utilities are intentionally independent of the core accident detection
logic so they can be reused throughout the application.

Covered Modules
---------------
- image
- time
- visualization

Testing Goals
-------------
- Verify utility function correctness.
- Verify image processing behavior.
- Verify visualization helpers.
- Verify timing utilities.
- Verify edge-case handling.
- Ensure deterministic behavior.
- Ensure platform independence.

Future test modules include:

- test_image.py
- test_time.py
- test_visualization.py

Design Principles
-----------------
- Fast execution
- Independent tests
- No external resources
- High code coverage
- Deterministic results

The utilities test package intentionally contains no runtime logic.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations