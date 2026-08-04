"""
Unit tests for road_accident_detection.core.paths.

These tests verify the project's path management utilities, ensuring that
directory creation, path resolution, and project structure handling behave
correctly across supported platforms.

Author:
    RoadAccidentAI Research Project

License:
    MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from road_accident_detection.core.paths import (
    ProjectPaths,
    ensure_directory,
    get_project_root,
    resolve_path,
)


def test_get_project_root_returns_existing_directory() -> None:
    """
    Verify that the project root exists.

    The returned path should exist and be a directory.
    """

    project_root = get_project_root()

    assert isinstance(project_root, Path)
    assert project_root.exists()
    assert project_root.is_dir()


def test_ensure_directory_creates_directory(
    tmp_path: Path,
) -> None:
    """
    Verify directory creation.

    Args:
        tmp_path:
            Temporary directory supplied by pytest.
    """

    directory = tmp_path / "new_directory"

    assert not directory.exists()

    returned = ensure_directory(directory)

    assert directory.exists()
    assert directory.is_dir()
    assert returned == directory


def test_ensure_directory_existing_directory(
    tmp_path: Path,
) -> None:
    """
    Verify ensure_directory works for an existing directory.

    Args:
        tmp_path:
            Temporary directory.
    """

    returned = ensure_directory(tmp_path)

    assert returned == tmp_path
    assert tmp_path.exists()
    assert tmp_path.is_dir()


def test_resolve_path_absolute(
    tmp_path: Path,
) -> None:
    """
    Verify absolute path resolution.

    Args:
        tmp_path:
            Temporary directory.
    """

    resolved = resolve_path(tmp_path)

    assert resolved == tmp_path.resolve()


def test_resolve_path_relative() -> None:
    """
    Verify relative path resolution.
    """

    relative = Path("configs")

    resolved = resolve_path(relative)

    assert resolved.is_absolute()


def test_project_paths_initialization() -> None:
    """
    Verify ProjectPaths construction.
    """

    paths = ProjectPaths()

    assert isinstance(paths.root, Path)
    assert isinstance(paths.configs, Path)
    assert isinstance(paths.datasets, Path)
    assert isinstance(paths.models, Path)
    assert isinstance(paths.outputs, Path)
    assert isinstance(paths.logs, Path)


def test_project_paths_root_exists() -> None:
    """
    Verify the project root exists.
    """

    paths = ProjectPaths()

    assert paths.root.exists()
    assert paths.root.is_dir()


def test_project_paths_children() -> None:
    """
    Verify child paths are Path objects.
    """

    paths = ProjectPaths()

    child_paths = (
        paths.configs,
        paths.datasets,
        paths.logs,
        paths.models,
        paths.outputs,
    )

    for path in child_paths:
        assert isinstance(path, Path)


@pytest.mark.parametrize(
    "directory_name",
    [
        "configs",
        "datasets",
        "logs",
        "models",
        "outputs",
    ],
)
def test_project_paths_directory_names(
    directory_name: str,
) -> None:
    """
    Verify standard directory names.

    Args:
        directory_name:
            Expected directory name.
    """

    paths = ProjectPaths()

    mapping = {
        "configs": paths.configs,
        "datasets": paths.datasets,
        "logs": paths.logs,
        "models": paths.models,
        "outputs": paths.outputs,
    }

    assert mapping[directory_name].name == directory_name


def test_resolve_path_returns_path_object() -> None:
    """
    Verify resolve_path always returns Path.
    """

    result = resolve_path("configs")

    assert isinstance(result, Path)