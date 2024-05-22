"""
Global-mean extension handling
"""

from __future__ import annotations

from pathlib import Path

from local.config.base import Config


def get_global_mean_supplement_files(gas: str, config: Config) -> list[Path]:
    """
    Get global-mean supplement files for a given gas

    Parameters
    ----------
    gas
        Gas

    config
        Configuration instance to use for this retrieval

    Returns
    -------
        Global-mean supplement files
    """
    return []
