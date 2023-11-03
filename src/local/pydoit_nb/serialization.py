"""
Serialization tools
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
U = TypeVar("U")


class ConfigBundleLike(Protocol[T]):
    """
    Protocol for configuration bundles
    """

    config_hydrated_path: Path
    """Path in which to write the hydrated config"""

    config_hydrated: T
    """Config to be hydrated"""


class Converter(Protocol[T_contra]):
    """
    Protocol for converters
    """

    def dumps(self, config: T_contra) -> str:
        """
        Dump config to a string
        """
        ...


def write_config_bundle_to_disk(
    config_bundle: ConfigBundleLike[U],
    converter: Converter[U],
) -> Path:
    """
    Write configuration bundle to disk

    Parameters
    ----------
    config_bundle
        Configuration bundle to write to disk

    converter
        Object that can serialize the configuration bundle's hydrated config

    Returns
    -------
        Path in which the configuration was written
    """
    write_path = config_bundle.config_hydrated_path
    with open(write_path, "w") as fh:
        fh.write(converter.dumps(config_bundle.config_hydrated))

    return write_path
