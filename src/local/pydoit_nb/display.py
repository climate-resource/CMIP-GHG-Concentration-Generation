"""
Utilities for displaying things, typically by printing
"""
from __future__ import annotations

from typing import Any

from .typing import ConfigBundleLike


def print_config_bundle(cb: ConfigBundleLike[Any]) -> None:
    """
    Print configuration bundle info

    Parameters
    ----------
    cb
        Config bundle
    """
    print(
        f"Will run {cb.config_id!r} with bundle serialised "
        f"in: {cb.config_hydrated_path!r}"
    )
