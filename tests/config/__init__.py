"""
Configuration test package for RoadAccidentAI.

This package contains unit tests for the project's configuration subsystem.

The configuration layer is responsible for loading, validating, and managing
application settings from YAML files and environment variables. Because every
component of the application depends on configuration, these tests help
ensure reproducible experiments and reliable runtime behavior.

Covered Modules
---------------
- defaults
- schemas
- settings
- loader

Testing Goals
-------------
- Verify configuration loading.
- Verify schema validation.
- Verify default values.
- Verify environment variable overrides.
- Verify configuration error handling.
- Ensure reproducible configuration behavior.

Future test modules include:

- test_defaults.py
- test_schemas.py
- test_settings.py
- test_loader.py

The configuration test package intentionally contains no runtime logic.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations