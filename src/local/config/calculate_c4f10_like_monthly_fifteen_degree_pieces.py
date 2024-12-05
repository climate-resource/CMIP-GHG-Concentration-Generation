"""
Config for the calculation of the 15 degree monthly data for gases we handle like C4F10
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class CalculateC4F10LikeMonthlyFifteenDegreePieces:
    """
    Configuration for the calculation of the 15 degree monthly data pieces for gases we handle like C4F10
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas to which this config applies (a bit redundant, but handy to be explicit)"""

    latitudinal_gradient_allyears_pcs_eofs_file: Path
    """
    Path in which to save the latitudinal gradient information for all years

    This contains the PCs and EOFs separately,
    but the PCs have been extended to cover all the years of interest.
    """

    latitudinal_gradient_pc0_total_emissions_regression_file: Path
    """
    Path in which to save the regression between pc0 and global emissions of the gas
    """

    global_annual_mean_allyears_file: Path
    """Path in which to save the global-, annual-mean, extended over all years"""

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
