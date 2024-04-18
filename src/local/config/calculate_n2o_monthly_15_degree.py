"""
Config for the calculation of the 15 degree monthly data for N2O
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class CalculateN2OMonthly15DegreeConfig:
    """
    Configuration class for the calculation of the 15 degree monthly data for N2O
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas to which this config applies (a bit redundant, but handy to be explicit)"""

    processed_bin_averages_file: Path
    """Path in which to save the spatial bin averages from the observational networks"""

    processed_data_file: Path
    """Path in which to save the processed, gridded data"""
