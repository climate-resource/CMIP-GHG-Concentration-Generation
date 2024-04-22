"""
Config for the calculation of the 15 degree monthly data for CH4
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class CalculateCH4Monthly15DegreeConfig:
    """
    Configuration class for the calculation of the 15 degree monthly data for CH4
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

    latitudinal_gradient_eofs_extended_file: Path
    """Path in which to save the extended latitudinal gradient EOFs and PCs"""

    latitudinal_gradient_file: Path
    """Path in which to save the latitudinal gradient"""

    seasonality_file: Path
    """Path in which to save the seasonality"""

    global_annual_mean_monthly_file: Path
    """Path in which to save the global-, annual-mean on a (year, month) time axis"""

    processed_data_file: Path
    """Path in which to save the processed, gridded data"""
