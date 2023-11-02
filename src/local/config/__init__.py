"""
Configuration handling
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Config, ConfigBundle


def get_config_bundles(
    root_dir_output: Path,
    run_id: str,
) -> list[ConfigBundle]:
    """
    Get configuration bundles

    All sorts of logic can be put in here. This is a very simple example.

    Parameters
    ----------
    root_dir_output
        Root directory in which output should be saved

    run_id
        ID for the run

    Returns
    -------
        Hydrated configuration bundles
    """
    configs = [
        Config(name="no-cov", covariance=np.array([[0.25, 0], [0, 0.5]])),
        Config(name="cov", covariance=np.array([[0.25, 0.5], [0.5, 0.5]])),
    ]

    bundles = [
        ConfigBundle(
            run_id=run_id,
            root_dir_output=root_dir_output,
            config_id=c.name,
            config_hydrated=c,
            config_hydrated_path=root_dir_output / c.name,
        )
        for c in configs
    ]

    return bundles
