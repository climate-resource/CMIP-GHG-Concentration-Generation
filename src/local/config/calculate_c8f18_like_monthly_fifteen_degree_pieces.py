"""
Config for the calculation of the 15 degree monthly data for gases we handle like C8F18
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class CalculateC8F18LikeMonthlyFifteenDegreePieces:
    """
    Configuration for the calculation of the 15 degree monthly data pieces for gases we handle like C8F18
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas to which this config applies (a bit redundant, but handy to be explicit)"""

    global_annual_mean_allyears_monthly_file: Path
    """
    Path in which to save the global-, annual-mean, interpolated to monthly steps for all years
    """

    seasonality_allyears_fifteen_degree_monthly_file: Path
    """
    Path for the seasonality on a 15 degree grid, interpolated to monthly steps for all years
    """

    latitudinal_gradient_fifteen_degree_allyears_monthly_file: Path
    """
    Path for the latitudinal gradient on a 15 degree grid, interpolated to monthly steps for all years
    """
