"""
Serialization tools
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from .typing import ConfigBundleLike, Converter

T = TypeVar("T")
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


def load_config_from_file(
    config_file: Path, target: type[T], converter: Converter[U]
) -> T:
    with open(config_file) as fh:
        config = converter.loads(fh.read(), target)

    return config
