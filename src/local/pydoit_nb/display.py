"""
Utilities for displaying things, typically by printing
"""
from __future__ import annotations

from typing import Any


def print_config(**kwargs: Any) -> None:
    """
    Print configuration

    Parameters
    ----------
    **kwargs
        Config to show
    """
    config_str = "\n".join([f"\t{k}: {v}" for k, v in kwargs.items()])
    print(f"Will run with the following config:\n{config_str}\n")
