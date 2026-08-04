"""
Model test package for RoadAccidentAI.

This package contains unit tests for the project's domain models.

The domain model layer defines the core data structures exchanged between
the pipeline, detector, visualization, and future research modules.
These models should remain framework-independent and represent the
application's internal data contracts.

Covered Modules
---------------
- frame
- video
- detection
- vehicle

Testing Goals
-------------
- Verify dataclass initialization.
- Verify default values.
- Verify property behavior.
- Verify serialization methods.
- Verify equality semantics.
- Verify validation logic.
- Ensure model independence.
- Ensure backward-compatible APIs.

Future test modules include:

- test_frame.py
- test_video.py
- test_detection.py
- test_vehicle.py

Design Principles
-----------------
- Fast execution
- Deterministic behavior
- High coverage
- No external dependencies
- Independent tests

The models test package intentionally contains no runtime logic.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations