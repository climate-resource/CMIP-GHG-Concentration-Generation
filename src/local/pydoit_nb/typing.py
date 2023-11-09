"""
Typing specifications
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypeVar

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)

DoitTaskSpec: TypeAlias = dict[str, Any]


class ConfigBundleLike(Protocol[T]):
    """
    Protocol for configuration bundles
    """

    config_hydrated_path: Path
    """Path in which to write the hydrated config"""

    config_hydrated: T
    """Config to be hydrated"""

    root_dir_output_run: Path
    """Root directory in which output is saved"""


class Converter(Protocol[T_contra]):
    """
    Protocol for converters
    """

    def dumps(self, config: T_contra, sort_keys: bool = False) -> str:
        """
        Dump configuration to a string

        Parameters
        ----------
        config
            Configuration to dump

        sort_keys
            Should the keys be sorted in the output?

        Returns
        -------
            String version of ``config``
        """
        ...  # pragma: no cover

    def loads(self, inp: str, target: type[T]) -> T:
        """
        Load an instance of ``target`` from a string

        Parameters
        ----------
        inp
            String to load from

        target
            Object type to return

        Returns
        -------
            Loaded instance of ``target``
        """
        ...  # pragma: no cover


HandleableConfiguration: TypeAlias = str
"""Config which we can handle and pass to a notebook via papermill"""


class NotebookConfigLike(Protocol):
    """
    A class which is like a notebook config
    """

    branch_config_id: str
    """String which identifies the branch config to use with the notebook"""
