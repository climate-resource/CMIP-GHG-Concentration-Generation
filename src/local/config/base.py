"""
Base configuration classes
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen

from .analysis import AnalysisConfig
from .constraint import ConstraintConfig
from .covariance import CovarianceConfig
from .covariance_plotting import CovariancePlottingConfig
from .figures import FiguresConfig
from .preparation import PreparationConfig
from .process import ProcessConfig
from .quick_crunch import QuickCrunchConfig
from .retrieve import RetrieveConfig


@frozen
class Config:
    """
    Configuration class used across all notebooks

    This is the key communication class between our notebooks. It must be used
    for passing all parameters into the notebooks via papermill.
    """

    name: str
    """Name of the configuration"""

    retrieve: list[RetrieveConfig]
    """Configurations to use with the retrieve branch"""
    # TODO: add validation that these all have unique branch_config_id

    process: list[ProcessConfig]
    """Configurations to use with the process branch"""
    # TODO: add validation that these all have unique branch_config_id

    quick_crunch: list[QuickCrunchConfig]
    """Configurations to use with the quick crunch branch"""
    # TODO: add validation that these all have unique branch_config_id

    preparation: list[PreparationConfig]
    """Configurations to use with the preparation step"""
    # TODO: add validation that these all have unique step_config_id

    covariance: list[CovarianceConfig]
    """Configurations to use with the covariance step"""
    # TODO: add validation that these all have unique step_config_id

    covariance_plotting: list[CovariancePlottingConfig]
    """Configuration to use for the quick plots of the covariance draws"""
    # TODO: add validation that these all have unique step_config_id

    constraint: list[ConstraintConfig]
    """Configurations to use with the constraint step"""
    # TODO: add validation that these all have unique step_config_id

    analysis: list[AnalysisConfig]
    """Configurations to use with the analysis step"""
    # TODO: add validation that these all have unique step_config_id

    figures: list[FiguresConfig]
    """Configurations to use with the figures step"""
    # TODO: add validation that these all have unique step_config_id


@frozen
class ConfigBundle:
    """
    Configuration bundle

    Has all key components in one place
    """

    run_id: str
    """ID for the run"""

    config_hydrated: Config
    """Hydrated config"""

    config_hydrated_path: Path
    """Path in/from which to read/write ``config_hydrated``"""

    root_dir_output: Path
    """Root output directory"""
    # TODO: add validation here that this is an absolute path and exists

    root_dir_output_run: Path
    """Root output directory for this run"""
    # TODO: add validation here that this is an absolute path and exists
    # TODO: add validation that this is a sub-directory of root_dir_output
