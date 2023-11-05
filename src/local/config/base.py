"""
Base configuration classes
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen

from .constraint import ConstraintConfig
from .covariance import CovarianceConfig
from .covariance_plotting import CovariancePlottingConfig
from .figures import FiguresConfig
from .preparation import PreparationConfig


@frozen
class Config:
    """
    Configuration class used across all notebooks

    This is the key communication class between our notebooks. It must be used
    for passing all parameters into the notebooks via papermill.
    """

    name: str
    """Name of the configuration"""

    preparation: list[PreparationConfig]
    """Configurations to use with the preparation branch"""
    # TODO: add validation that these all have unique branch_config_id

    covariance: list[CovarianceConfig]
    """Configurations to use with the covariance branch"""
    # TODO: add validation that these all have unique branch_config_id

    covariance_plotting: list[CovariancePlottingConfig]
    """Configuration to use for the quick plots of the covariance draws"""
    # TODO: add validation that these all have unique branch_config_id

    constraint: list[ConstraintConfig]
    """Configurations to use with the constraint branch"""
    # TODO: add validation that these all have unique branch_config_id

    figures: list[FiguresConfig]
    """Configurations to use with the figures branch"""
    # TODO: add validation that these all have unique branch_config_id


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

    output_notebook_dir: Path
    """Directory in which to write out the notebooks"""
    # TODO: decide whether to force this to be a sub-directory of
    # root_dir_output or not
