"""
Config for smoothing the Ghosh et al. 2023 data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class SmoothGhoshEtAl2023DataConfig:
    """
    Configuration class for smoothing of the Ghosh et al. 2023 data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """
    smoothed_file: Path
    """Path in which to save the smoothed timeseries"""
