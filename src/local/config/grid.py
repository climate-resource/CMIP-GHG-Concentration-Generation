"""
Config for the gridding step
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class GridConfig:
    """
    Configuration class for the gridding step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    processed_data_file: Path
    """Path in which to save the processed gridded data"""
