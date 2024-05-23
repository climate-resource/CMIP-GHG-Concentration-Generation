"""
Create config for crunching 15 degree pieces/inputs
"""

from __future__ import annotations

from pathlib import Path

import pint

from local.config.calculate_ch4_monthly_15_degree import (
    CalculateCH4MonthlyFifteenDegreePieces,
)
from local.config.calculate_co2_monthly_15_degree import (
    CalculateCO2MonthlyFifteenDegreePieces,
)
from local.config.calculate_n2o_monthly_15_degree import (
    CalculateN2OMonthlyFifteenDegreePieces,
)
from local.config.calculate_sf6_like_monthly_15_degree import (
    CalculateSF6LikeMonthlyFifteenDegreePieces,
    SF6LikePreIndustrialConfig,
)

Q = pint.get_application_registry().Quantity  # type: ignore


PieceCalculationOption = (
    CalculateCH4MonthlyFifteenDegreePieces
    | CalculateCO2MonthlyFifteenDegreePieces
    | CalculateN2OMonthlyFifteenDegreePieces
    | CalculateSF6LikeMonthlyFifteenDegreePieces
)


def create_monthly_fifteen_degree_pieces_configs(  # noqa: PLR0912
    gases: tuple[str, ...],
    gases_long_poleward_extension: tuple[str, ...] = (),
    gases_drop_obs_data_years_before_inclusive: dict[str, int] | None = None,
    gases_drop_obs_data_years_after_inclusive: dict[str, int] | None = None,
) -> dict[str, list[PieceCalculationOption],]:
    """
    Create configuration for calculating the monthly, 15 degree pieces for different gases

    Parameters
    ----------
    gases
        Gases for which to create the configuration

    gases_long_poleward_extension
        Gases for which we allow a long poleward extension of data.

    gases_drop_obs_data_years_before_inclusive
        Years before which to drop observational data (inclusive) for gases.
        If a gas is not in the list, no drop year will be applied.

    gases_drop_obs_data_years_after_inclusive
        Years after which to drop observational data (inclusive) for gases.
        If a gas is not in the list, no drop year will be applied.

    Returns
    -------
        Configuration for calculating the monthly, 15 degree pieces for each gas
        in ``gases``
    """
    if gases_drop_obs_data_years_before_inclusive is None:
        gases_drop_obs_data_years_before_inclusive = {}

    if gases_drop_obs_data_years_after_inclusive is None:
        gases_drop_obs_data_years_after_inclusive = {}

    out: dict[str, list[PieceCalculationOption]] = {}

    for gas in gases:
        if gas == "co2":
            out["calculate_co2_monthly_fifteen_degree_pieces"] = [
                get_co2_monthly_fifteen_degree_pieces_config()
            ]

        elif gas == "ch4":
            out["calculate_ch4_monthly_fifteen_degree_pieces"] = [
                get_ch4_monthly_fifteen_degree_pieces_config()
            ]

        elif gas == "n2o":
            out["calculate_n2o_monthly_fifteen_degree_pieces"] = [
                get_n2o_monthly_fifteen_degree_pieces_config()
            ]

        elif gas in (
            "cfc11",
            "cfc113",
            "cfc114",
            "cfc115",
            "cfc12",
            "ch2cl2",
            "ch3br",
            "ch3ccl3",
            "ch3cl",
            "chcl3",
            "halon1211",
            "halon1301",
            "halon2402",
            "hcfc141b",
            "hcfc142b",
            "hcfc22",
            "hfc125",
            "hfc134a",
            "hfc143a",
            "hfc152a",
            # Up to here
            "sf6",
        ):
            if "calculate_sf6_like_monthly_fifteen_degree_pieces" not in out:
                out["calculate_sf6_like_monthly_fifteen_degree_pieces"] = []

            if gas in gases_drop_obs_data_years_before_inclusive:
                year_drop_observational_data_before_and_including: int | None = (
                    gases_drop_obs_data_years_before_inclusive[gas]
                )
            else:
                year_drop_observational_data_before_and_including = None

            if gas in gases_drop_obs_data_years_after_inclusive:
                year_drop_observational_data_after_and_including: int | None = (
                    gases_drop_obs_data_years_after_inclusive[gas]
                )
            else:
                year_drop_observational_data_after_and_including = None

            out["calculate_sf6_like_monthly_fifteen_degree_pieces"].append(
                get_sf6_like_monthly_fifteen_degree_pieces_config(
                    gas=gas,
                    allow_long_poleward_extension=gas in gases_long_poleward_extension,
                    year_drop_observational_data_before_and_including=year_drop_observational_data_before_and_including,
                    year_drop_observational_data_after_and_including=year_drop_observational_data_after_and_including,
                )
            )

        else:
            raise NotImplementedError(gas)

    return out


def get_ch4_monthly_fifteen_degree_pieces_config() -> (
    CalculateCH4MonthlyFifteenDegreePieces
):
    """
    Get the configuration for calculating the monthly, 15 degree pieces for CH4

    Returns
    -------
        Configuration for calculating the monthly, 15 degree pieces for CH4
    """
    interim_dir = Path("data/interim/ch4")

    return CalculateCH4MonthlyFifteenDegreePieces(
        step_config_id="only",
        gas="ch4",
        processed_bin_averages_file=interim_dir
        / "ch4_observational-network_bin-averages.csv",
        observational_network_interpolated_file=interim_dir
        / "ch4_observational-network_interpolated.nc",
        observational_network_global_annual_mean_file=interim_dir
        / "ch4_observational-network_global-annual-mean.nc",
        lat_gradient_n_eofs_to_use=2,
        observational_network_latitudinal_gradient_eofs_file=interim_dir
        / "ch4_observational-network_latitudinal-gradient-eofs.nc",
        observational_network_seasonality_file=interim_dir
        / "ch4_observational-network_seasonality.nc",
        latitudinal_gradient_allyears_pcs_eofs_file=interim_dir
        / "ch4_allyears-lat-gradient-eofs-pcs.nc",
        latitudinal_gradient_pc0_ch4_fossil_emissions_regression_file=interim_dir
        / "ch4_pc0-ch4-fossil-emissions-regression.yaml",
        global_annual_mean_allyears_file=interim_dir
        / "ch4_global-annual-mean_allyears.nc",
        global_annual_mean_allyears_monthly_file=interim_dir
        / "ch4_global-annual-mean_allyears-monthly.nc",
        seasonality_allyears_fifteen_degree_monthly_file=interim_dir
        / "ch4_seasonality_fifteen-degree_allyears-monthly.nc",
        latitudinal_gradient_fifteen_degree_allyears_monthly_file=interim_dir
        / "ch4_latitudinal-gradient_fifteen-degree_allyears-monthly.nc",
    )


def get_n2o_monthly_fifteen_degree_pieces_config() -> (
    CalculateN2OMonthlyFifteenDegreePieces
):
    """
    Get the configuration for calculating the monthly, 15 degree pieces for N2O

    Returns
    -------
        Configuration for calculating the monthly, 15 degree pieces for N2O
    """
    interim_dir = Path("data/interim/n2o")

    return CalculateN2OMonthlyFifteenDegreePieces(
        step_config_id="only",
        gas="n2o",
        processed_bin_averages_file=interim_dir
        / "n2o_observational-network_bin-averages.csv",
        observational_network_interpolated_file=interim_dir
        / "n2o_observational-network_interpolated.nc",
        observational_network_global_annual_mean_file=interim_dir
        / "n2o_observational-network_global-annual-mean.nc",
        lat_gradient_n_eofs_to_use=2,
        observational_network_latitudinal_gradient_eofs_file=interim_dir
        / "n2o_observational-network_latitudinal-gradient-eofs.nc",
        observational_network_seasonality_file=interim_dir
        / "n2o_observational-network_seasonality.nc",
        latitudinal_gradient_allyears_pcs_eofs_file=interim_dir
        / "n2o_allyears-lat-gradient-eofs-pcs.nc",
        latitudinal_gradient_pc0_n2o_emissions_regression_file=interim_dir
        / "n2o_pc0-n2o-fossil-emissions-regression.yaml",
        global_annual_mean_allyears_file=interim_dir
        / "n2o_global-annual-mean_allyears.nc",
        global_annual_mean_allyears_monthly_file=interim_dir
        / "n2o_global-annual-mean_allyears-monthly.nc",
        seasonality_allyears_fifteen_degree_monthly_file=interim_dir
        / "n2o_seasonality_fifteen-degree_allyears-monthly.nc",
        latitudinal_gradient_fifteen_degree_allyears_monthly_file=interim_dir
        / "n2o_latitudinal-gradient_fifteen-degree_allyears-monthly.nc",
    )


def get_co2_monthly_fifteen_degree_pieces_config() -> (
    CalculateCO2MonthlyFifteenDegreePieces
):
    """
    Get the configuration for calculating the monthly, 15 degree pieces for CO2

    Returns
    -------
        Configuration for calculating the monthly, 15 degree pieces for CO2
    """
    interim_dir = Path("data/interim/co2")

    return CalculateCO2MonthlyFifteenDegreePieces(
        step_config_id="only",
        gas="co2",
        processed_bin_averages_file=interim_dir
        / "co2_observational-network_bin-averages.csv",
        observational_network_interpolated_file=interim_dir
        / "co2_observational-network_interpolated.nc",
        observational_network_global_annual_mean_file=interim_dir
        / "co2_observational-network_global-annual-mean.nc",
        lat_gradient_n_eofs_to_use=2,
        observational_network_latitudinal_gradient_eofs_file=interim_dir
        / "co2_observational-network_latitudinal-gradient-eofs.nc",
        observational_network_seasonality_file=interim_dir
        / "co2_observational-network_seasonality.nc",
        seasonality_change_n_eofs_to_use=1,
        observational_network_seasonality_change_eofs_file=interim_dir
        / "co2_observational-network_seasonality-change-eofs.nc",
        latitudinal_gradient_allyears_pcs_eofs_file=interim_dir
        / "co2_allyears-lat-gradient-eofs-pcs.nc",
        latitudinal_gradient_pc0_co2_fossil_emissions_regression_file=interim_dir
        / "co2_pc0-co2-fossil-emissions-regression.yaml",
        seasonality_change_allyears_pcs_eofs_file=interim_dir
        / "co2_allyears-seasonality-change-eofs-pcs.nc",
        seasonality_change_temperature_co2_conc_regression_file=interim_dir
        / "co2_seasonality-change_temp-conc-regression.yaml",
        global_annual_mean_allyears_file=interim_dir
        / "co2_global-annual-mean_allyears.nc",
        global_annual_mean_allyears_monthly_file=interim_dir
        / "co2_global-annual-mean_allyears-monthly.nc",
        seasonality_allyears_fifteen_degree_monthly_file=interim_dir
        / "co2_seasonality_fifteen-degree_allyears-monthly.nc",
        latitudinal_gradient_fifteen_degree_allyears_monthly_file=interim_dir
        / "co2_latitudinal-gradient_fifteen-degree_allyears-monthly.nc",
    )


PRE_INDUSTRIAL_VALUES_DEFAULT = {
    "cfc11": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1950, source="Guessing from reading M2017"
    ),
    "cfc113": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1930, source="Guessing from reading M2017"
    ),
    "cfc114": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1940, source="Guessing from reading M2017"
    ),
    "cfc115": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1960, source="Guessing from reading M2017"
    ),
    "cfc12": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1940, source="Guessing from reading M2017"
    ),
    "ch2cl2": SF6LikePreIndustrialConfig(
        value=Q(8.0, "ppt"), year=1940, source="Guessing from reading M2017"
    ),
    "ch3br": SF6LikePreIndustrialConfig(
        value=Q(5.5, "ppt"), year=1925, source="Guessing from reading M2017"
    ),
    "ch3ccl3": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1955, source="Guessing from reading M2017"
    ),
    "ch3cl": SF6LikePreIndustrialConfig(
        value=Q(460.0, "ppt"), year=1940, source="Guessing from reading M2017"
    ),
    "chcl3": SF6LikePreIndustrialConfig(
        value=Q(6.0, "ppt"), year=1940, source="Guessing from reading M2017"
    ),
    "halon1211": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1965, source="Guessing from reading M2017"
    ),
    "halon1301": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1965, source="Guessing from reading M2017"
    ),
    "halon2402": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1965, source="Guessing from reading M2017"
    ),
    "hcfc141b": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1990, source="Guessing from reading M2017"
    ),
    "hcfc142b": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1970, source="Guessing from reading M2017"
    ),
    "hcfc22": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1955, source="Guessing from reading M2017"
    ),
    "hfc125": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1990, source="Guessing from reading M2017"
    ),
    "hfc134a": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1990, source="Guessing from reading M2017"
    ),
    "hfc143a": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1975, source="Guessing from reading M2017"
    ),
    "hfc152a": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1985, source="Guessing from reading M2017"
    ),
    # up to here
    "sf6": SF6LikePreIndustrialConfig(
        value=Q(0.0, "ppt"), year=1950, source="Guessing from reading M2017"
    ),
}
"""Default values to use for pre-industrial"""


def get_sf6_like_monthly_fifteen_degree_pieces_config(  # noqa: PLR0913
    gas: str,
    pre_industrial: SF6LikePreIndustrialConfig | None = None,
    allow_poleward_extension: bool = True,
    allow_long_poleward_extension: bool = False,
    year_drop_observational_data_before_and_including: int | None = None,
    year_drop_observational_data_after_and_including: int | None = None,
) -> CalculateSF6LikeMonthlyFifteenDegreePieces:
    """
    Get the configuration for calculating the monthly, 15 degree pieces for a gas handled like SF6

    Parameters
    ----------
    gas
        Gas for which to create the config

    pre_industrial
        Pre-industrial value.
        If not supplied, we use the value from {py:const}`PRE_INDUSTRIAL_VALUES_DEFAULT`
        for ``gas``.

    allow_poleward_extension
        Allow poleward extension of the data over one latitudinal bin.

    allow_long_poleward_extension
        Allow poleward extension of the data over multiple latitudinal bins.

    year_drop_observational_data_before_and_including
        Year (inclusive) before which to drop observational data.
        This helps us deal with data gaps.

    year_drop_observational_data_after_and_including
        Year (inclusive) after which to drop observational data.
        This helps us deal with data gaps.

    Returns
    -------
        Configuration for calculating the monthly, 15 degree pieces for a gas handled like SF6
    """
    interim_dir = Path(f"data/interim/{gas}")

    if pre_industrial is None:
        pre_industrial = PRE_INDUSTRIAL_VALUES_DEFAULT[gas]

    return CalculateSF6LikeMonthlyFifteenDegreePieces(
        step_config_id=gas,
        gas=gas,
        pre_industrial=pre_industrial,
        processed_bin_averages_file=interim_dir
        / f"{gas}_observational-network_bin-averages.csv",
        processed_all_data_with_bins_file=interim_dir
        / f"{gas}_observational-network_all-data-with-bin-information.csv",
        allow_poleward_extension=allow_poleward_extension,
        allow_long_poleward_extension=allow_long_poleward_extension,
        observational_network_interpolated_file=interim_dir
        / f"{gas}_observational-network_interpolated.nc",
        observational_network_global_annual_mean_file=interim_dir
        / f"{gas}_observational-network_global-annual-mean.nc",
        year_drop_observational_data_before_and_including=year_drop_observational_data_before_and_including,
        year_drop_observational_data_after_and_including=year_drop_observational_data_after_and_including,
        lat_gradient_n_eofs_to_use=1,
        observational_network_latitudinal_gradient_eofs_file=interim_dir
        / f"{gas}_observational-network_latitudinal-gradient-eofs.nc",
        observational_network_seasonality_file=interim_dir
        / f"{gas}_observational-network_seasonality.nc",
        latitudinal_gradient_allyears_pcs_eofs_file=interim_dir
        / f"{gas}_allyears-lat-gradient-eofs-pcs.nc",
        latitudinal_gradient_pc0_total_emissions_regression_file=interim_dir
        / f"{gas}_pc0-total-emissions-regression.yaml",
        global_annual_mean_allyears_file=interim_dir
        / f"{gas}_global-annual-mean_allyears.nc",
        global_annual_mean_allyears_monthly_file=interim_dir
        / f"{gas}_global-annual-mean_allyears-monthly.nc",
        seasonality_allyears_fifteen_degree_monthly_file=interim_dir
        / f"{gas}_seasonality_fifteen-degree_allyears-monthly.nc",
        latitudinal_gradient_fifteen_degree_allyears_monthly_file=interim_dir
        / f"{gas}_latitudinal-gradient_fifteen-degree_allyears-monthly.nc",
    )
