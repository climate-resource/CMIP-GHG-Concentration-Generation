"""
Typing specifications
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypeVar

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

DoitTaskSpec: TypeAlias = dict[str, Any]


class ConfigBundleLike(Protocol[T_co]):
    """
    Protocol for configuration bundles
    """

    @property
    def config_hydrated(self) -> T_co:
        """Hydrated config"""
        ...

    @property
    def root_dir_output_run(self) -> Path:
        """Root directory in which output is saved"""
        ...

    @property
    def config_hydrated_path(self) -> Path:
        """Path in which to write the hydrated config"""
        ...


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

    def loads(self, inp: str, target: type[T_co]) -> T_co:
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

    step_config_id: str
    """String which identifies the step config to use with the notebook"""
