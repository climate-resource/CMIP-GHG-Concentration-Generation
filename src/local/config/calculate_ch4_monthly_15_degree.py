"""
Config for the calculation of the 15 degree monthly data for CH4
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class CalculateCH4MonthlyFifteenDegreeHalfDegreeConfig:
    """
    Configuration class for the calculation of the 15 degree and 0.5 degree monthly data for CH4
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

    observational_network_interpolated_file: Path
    """Path in which to save the interpolated observational network data"""

    observational_network_global_annual_mean_file: Path
    """Path in which to save the global-mean of the observational network data"""

    lat_gradient_n_eofs_to_use: int
    """Number of EOFs to use for latitudinal gradient calculations"""

    observational_network_latitudinal_gradient_eofs_file: Path
    """Path in which to save the latitudinal gradient EOFs of the observational network data"""

    observational_network_seasonality_file: Path
    """Path in which to save the seasonality of the observational network data"""

    latitudinal_gradient_allyears_pcs_eofs_file: Path
    """
    Path in which to save the latitudinal gradient information for all years

    This contains the PCs and EOFs separately,
    but the PCs have been extended to cover all the years of interest.
    """

    latitudinal_gradient_pc0_ch4_fossil_emissions_regression_file: Path
    """
    Path in which to save the regression between pc0 and fossil CH4 emissions
    """

    global_annual_mean_allyears_file: Path
    """Path in which to save the global-, annual-mean, extended over all years"""

    global_annual_mean_allyears_monthly_file: Path
    """
    Path in which to save the global-, annual-mean, interpolated to monthly steps for all years
    """

    seasonality_allyears_monthly_file: Path
    """
    Path in which to save the seasonality, interpolated to monthly steps for all years
    """

    latitudinal_gradient_allyears_monthly_file: Path
    """
    Path in which to save the latitudinal gradient, interpolated to monthly steps for all years
    """

    latitudinal_gradient_half_degree_allyears_monthly_file: Path
    """
    Path in which to save the 0.5 degree latitudinal gradient, interpolated to monthly steps for all years
    """
