"""
Typing specifications
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypeVar

T = TypeVar("T")

DoitTaskSpec: TypeAlias = dict[str, Any]


class ConfigBundleLike(Protocol[T]):
    """
    Protocol for configuration bundles
    """

    config_id: str
    """ID for this set of config, unique among all configuration bundles"""

    config_hydrated_path: Path
    """Path in which to write the hydrated config"""

    config_hydrated: T
    """Config to be hydrated"""
