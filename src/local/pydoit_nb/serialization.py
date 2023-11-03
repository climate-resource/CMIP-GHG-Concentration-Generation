"""
Serialization tools
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from .typing import ConfigBundleLike, Converter

U = TypeVar("U")


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
