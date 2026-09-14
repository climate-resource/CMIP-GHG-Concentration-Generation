"""
Config for the calculation of the 15 degree monthly data for CO2
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class CalculateCO2MonthlyFifteenDegreePieces:
    """
    Configuration class for the calculation of the 15 degree monthly data pieces for CO2
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas to which this config applies (a bit redundant, but handy to be explicit)"""

    include_satellite_data: bool
    """Whether to include scaled satellite data on top of the ground-based observational network"""

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

    seasonality_change_n_eofs_to_use: int
    """Number of EOFs to use for seasonality change calculations"""

    observational_network_seasonality_change_eofs_file: Path
    """Path in which to save the seasonality change EOFs of the observational network data"""

    latitudinal_gradient_allyears_pcs_eofs_file: Path
    """
    Path in which to save the latitudinal gradient information for all years

    This contains the PCs and EOFs separately,
    but the PCs have been extended to cover all the years of interest.
    """

    latitudinal_gradient_pc0_co2_fossil_emissions_regression_file: Path
    """
    Path in which to save the regression between pc0 and fossil CO2 emissions
    """

    seasonality_change_allyears_pcs_eofs_file: Path
    """
    Path in which to save the seasonality change information for all years

    This contains the PCs and EOFs separately,
    but the PCs have been extended to cover all the years of interest.
    """

    seasonality_change_temperature_co2_conc_regression_file: Path
    """
    Path in which to save the regression between delta seasonality and the composite temp-conc timeseries
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

    satellite_fit: str | None
    """
    Fit used for the satellite data (e.g. ``"LINEAR_FIT"``), if ``include_satellite_data`` is ``True``

    Only used to label diagnostics files so that runs with different fits
    (or no satellite data at all) don't overwrite each other's diagnostics.
    """

    diagnostics_dir: Path
    """Directory in which to save diagnostics that let different runs be compared"""

    year_drop_observational_data_before_and_including: int | None = None
    """
    Year (inclusive) before which to drop observational-network data

    Applied to the ground-based observational network only - ice-core/firn
    data used to extend the record back in time is unaffected. If ``None``,
    no data is dropped.
    """

    test_split_fraction: float | None = None
    """
    Fraction of ground-based observational-network rows to randomly hold out as a test set

    Applied at the row level (one row is one station-month observation),
    before binning, so the held-out rows never influence the pipeline's
    output. If ``None``, no split is performed and all data is used.
    """

    test_split_seed: int | None = None
    """
    Seed used for the random train/test split, for reproducibility

    Only used if ``test_split_fraction`` is set.
    """

    held_out_test_data_file: Path | None = None
    """
    Path in which to save the held-out test rows from the train/test split

    Only written if ``test_split_fraction`` is set.
    """

    weight_satellite_data: bool = False
    """
    Whether to weight satellite data by its retrieval uncertainty when combining it with ground-based data

    Only used if ``include_satellite_data`` is ``True``. Ground-based
    stations keep an equal weight of one each; satellite bins are
    weighted by the inverse variance of ``xco2_scaled_stderr_rchi2``.
    If ``False``, ground and satellite data are combined exactly as
    before (an unweighted concatenation).
    """
