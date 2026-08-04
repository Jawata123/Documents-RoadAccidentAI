"""
Vision test package for RoadAccidentAI.

This package contains unit tests for the computer vision subsystem.

The vision layer is responsible for detector abstraction, model inference,
detection result normalization, and future integration with additional
computer vision backends. These tests ensure that every detector
implementation follows the same interface and produces consistent outputs.

Covered Modules
---------------
- detector.base
- detector.factory
- detector.result
- detector.ultralytics_yolo

Testing Goals
-------------
- Verify detector interface compliance.
- Verify detector factory behavior.
- Verify detection result correctness.
- Verify YOLO detector initialization.
- Verify inference pipeline behavior.
- Verify error handling.
- Ensure backend independence.

Future test modules include:

- detector/test_base.py
- detector/test_factory.py
- detector/test_result.py
- detector/test_ultralytics_yolo.py

Design Principles
-----------------
- Fast execution where possible.
- Independent unit tests.
- Deterministic behavior.
- High code coverage.
- Clear separation between unit and integration tests.

The vision test package intentionally contains no runtime logic.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations