"""
Base configuration classes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from attrs import frozen


@frozen
class Config:
    """
    Configuration class used across all notebooks

    This is the key communication class between our notebooks. It must be used
    for passing all parameters into the notebooks via papermill.
    """

    name: str
    """Name of the configuration"""

    covariance: npt.NDArray[np.float64]
    """Covariance to use when making draws"""


@frozen
class ConfigBundle:
    """
    Configuration bundle

    Has all key components in one place
    """

    run_id: str
    """ID for the run"""

    config_id: str
    """ID to identify this particular set of hydrated config, separate from all others"""

    config_hydrated: Config
    """Hydrated config"""

    config_hydrated_path: Path
    """Path in/from which to read/write ``config_hydrated``"""

    root_dir_output: Path
    """Root output directory"""
    # TODO: add validation here that this is an absolute path and exists
