"""
Config for the gridded data processing step
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class GriddedDataProcessingConfig:
    """
    Configuration class for the gridded data processing step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    processed_data_file_global_hemispheric_means: Path
    """Path in which to save the processed global-/hemispheric-mean data"""

    processed_data_file_global_hemispheric_annual_means: Path
    """Path in which to save the processed global-/hemispheric-, annual-mean data"""
