"""
Base configuration classes
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen

from .grid import GridConfig
from .gridded_data_processing import GriddedDataProcessingConfig
from .process import ProcessConfig
from .process_noaa import ProcessNOAADataConfig
from .quick_crunch import QuickCrunchConfig
from .retrieve import RetrieveConfig
from .write_input4mips import WriteInput4MIPsConfig


@frozen
class Config:
    """
    Configuration class used across all notebooks

    This is the key communication class between our notebooks. It must be used
    for passing all parameters into the notebooks via papermill.
    """

    name: str
    """Name of the configuration"""

    version: str
    """Version ID for this configuration"""
    # TODO: add validation that this matches semantic versioning

    ci: bool
    """
    Is this configuration for a CI run?

    We use this to help us create a short-cut path that can reasonably be run
    as part of our CI workflow.
    """

    process_noaa_data: list[ProcessNOAADataConfig]
    """Configurations to use for processing NOAA data"""
    # TODO: add validation that these all have unique step_config_id

    retrieve: list[RetrieveConfig]
    """Configurations to use with the retrieve step"""
    # TODO: add validation that these all have unique step_config_id

    process: list[ProcessConfig]
    """Configurations to use with the process step"""
    # TODO: add validation that these all have unique step_config_id

    grid: list[GridConfig]
    """Configurations to use with the grid step"""
    # TODO: add validation that these all have unique step_config_id

    gridded_data_processing: list[GriddedDataProcessingConfig]
    """Configurations to use with the gridded data processing step"""
    # TODO: add validation that these all have unique step_config_id

    write_input4mips: list[WriteInput4MIPsConfig]
    """Configurations to use with the write input4MIPs step"""
    # TODO: add validation that these all have unique step_config_id

    quick_crunch: list[QuickCrunchConfig]
    """Configurations to use with the quick crunch step"""
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
