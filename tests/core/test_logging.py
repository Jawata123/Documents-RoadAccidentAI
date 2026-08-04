"""
Unit tests for road_accident_detection.core.logging.

These tests verify the project's centralized logging utilities, ensuring
correct logger creation, singleton configuration behavior, and file logging.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

import logging
from pathlib import Path

from road_accident_detection.core.logging import (
    configure_logging,
    get_logger,
)


def test_get_logger_returns_logger() -> None:
    """
    Verify that get_logger() returns a Logger instance.
    """

    logger = get_logger()

    assert isinstance(
        logger,
        logging.Logger,
    )


def test_get_named_logger() -> None:
    """
    Verify that named loggers preserve their name.
    """

    logger = get_logger("unit_test_logger")

    assert logger.name == "unit_test_logger"


def test_same_logger_instance() -> None:
    """
    Verify that requesting the same logger twice returns
    the same Logger instance.
    """

    logger1 = get_logger("shared_logger")
    logger2 = get_logger("shared_logger")

    assert logger1 is logger2


def test_configure_logging_creates_log_directory(
    tmp_path: Path,
) -> None:
    """
    Verify that configure_logging creates the log directory.

    Args:
        tmp_path:
            Temporary directory provided by pytest.
    """

    log_directory = tmp_path / "logs"

    configure_logging(
        log_directory=log_directory,
    )

    assert log_directory.exists()
    assert log_directory.is_dir()


def test_configure_logging_creates_log_file(
    tmp_path: Path,
) -> None:
    """
    Verify that a log file is created after logging.

    Args:
        tmp_path:
            Temporary directory provided by pytest.
    """

    log_directory = tmp_path / "logs"

    configure_logging(
        log_directory=log_directory,
    )

    logger = get_logger("file_test")

    logger.info("Logging test message.")

    log_files = list(
        log_directory.glob("*.log")
    )

    assert len(log_files) == 1
    assert log_files[0].exists()


def test_logger_can_write_messages(
    caplog,
) -> None:
    """
    Verify that log messages are emitted.

    Args:
        caplog:
            Pytest log capture fixture.
    """

    logger = get_logger("message_test")

    with caplog.at_level(logging.INFO):
        logger.info("Hello RoadAccidentAI")

    assert "Hello RoadAccidentAI" in caplog.text


def test_multiple_named_loggers() -> None:
    """
    Verify independent named logger creation.
    """

    logger_a = get_logger("logger_a")
    logger_b = get_logger("logger_b")

    assert logger_a is not logger_b

    assert logger_a.name == "logger_a"
    assert logger_b.name == "logger_b"


def test_logger_level_is_valid() -> None:
    """
    Verify logger has a valid logging level.
    """

    logger = get_logger()

    assert logger.level in (
        logging.NOTSET,
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    )