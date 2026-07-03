"""
Config for processing scaled satellite data from OBS4MIPs
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class ProcessScaledSatDataConfig:
    """
    Configuration class for processing scaled satellite data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas for which we are processing data"""

    scaled_data_path: Path
