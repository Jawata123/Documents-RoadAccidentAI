"""
Detector test package for RoadAccidentAI.

This package contains unit tests for the detector abstraction layer used by
the RoadAccidentAI computer vision pipeline.

The detector package is responsible for providing a common interface for
object detection models while allowing different detection backends to be
integrated without affecting the rest of the application.

Covered Modules
---------------
- base
- factory
- result
- ultralytics_yolo

Testing Goals
-------------
- Verify detector interface compliance.
- Verify detector factory creation.
- Verify detection result containers.
- Verify detector initialization.
- Verify inference behavior.
- Verify error handling.
- Ensure detector backend independence.

Future test modules
-------------------
- test_base.py
- test_factory.py
- test_result.py
- test_ultralytics_yolo.py

Design Principles
-----------------
- Fast execution where practical.
- Independent and deterministic tests.
- High code coverage.
- Consistent behavior across detector implementations.
- Support future detector backends without API changes.

This package intentionally contains no executable runtime logic.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations