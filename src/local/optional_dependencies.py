"""
Handling of optional dependencies
"""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingOptionalDependencyError(ImportError):
    """
    Raised to signal that an optional dependency is not available
    """

    def __init__(
        self,
        dependency: str,
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        dependency
            Optional dependency that is missing.
        """
        error_msg = (
            f"{dependency} not available. "
            "Please install it using your package manager (e.g. pixi, poetry, pip, conda etc.)"
        )

        super().__init__(error_msg)


def get_optional_dependency(dependency: str) -> ModuleType:
    """
    Get an optional dependency

    Parameters
    ----------
    dependency
        Dependency to get

    Returns
    -------
    :
        Imported optional dependency

    Raises
    ------
    MissingOptionalDependencyError
        The dependency is not installed
    """
    try:
        return importlib.import_module(dependency)
    except ImportError as exc:
        raise MissingOptionalDependencyError(dependency) from exc
