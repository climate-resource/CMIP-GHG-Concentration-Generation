"""
Config for the grid crunching step
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class GridCrunchingConfig:
    """
    Configuration class for the grid crunching step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas for which we are crunching gridded data"""

    fifteen_degree_monthly_file: Path
    """Path in which to save the 15 degree, monthly gridded data"""

    # half_degree_monthly_file: Path
    # """Path in which to save the 0.5 degree, monthly gridded data"""

    global_mean_monthly_file: Path
    """Path in which to save the global-mean, monthly data"""

    hemispheric_mean_monthly_file: Path
    """Path in which to save the hemispheric-mean, monthly data"""

    global_mean_annual_mean_file: Path
    """Path in which to save the global-mean, annual-mean data"""

    hemispheric_mean_annual_mean_file: Path
    """Path in which to save the hemispheric-mean, annual-mean data"""
