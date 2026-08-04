"""
Pipeline test package for RoadAccidentAI.

This package contains unit tests for the application's processing pipeline.

The pipeline layer orchestrates the complete computer vision workflow,
connecting video sources, frame processing, object detection, and future
research modules while remaining independent of specific detector
implementations.

Covered Modules
---------------
- source
- frame
- runner

Testing Goals
-------------
- Verify video source management.
- Verify frame processing workflow.
- Verify pipeline execution.
- Verify detector integration.
- Verify error handling.
- Verify resource management.
- Ensure deterministic pipeline behavior.

Future test modules include:

- test_source.py
- test_frame.py
- test_runner.py

Design Principles
-----------------
- Independent unit tests.
- Fast execution.
- Deterministic behavior.
- No external network dependencies.
- Minimal filesystem interaction.
- High code coverage.

The pipeline package represents the orchestration layer between the
vision subsystem and future research components such as tracking,
trajectory analysis, accident reasoning, event generation, and alerting.

This package intentionally contains no executable runtime logic.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations