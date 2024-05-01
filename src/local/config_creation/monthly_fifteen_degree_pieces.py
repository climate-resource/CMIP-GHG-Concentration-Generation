"""
Create config for crunching 15 degree pieces/inputs
"""

from __future__ import annotations

from local.config.calculate_ch4_monthly_15_degree import (
    CalculateCH4MonthlyFifteenDegreePieces,
)


def create_monthly_fifteen_degree_pieces_configs(
    gases: tuple[str, ...],
) -> dict[str, CalculateCH4MonthlyFifteenDegreePieces]:
    out = {}

    for gas in gases:
        if gas == "ch4":
            out["calculate_ch4_monthly_fifteen_degree_pieces"] = [
                get_ch4_monthly_fifteen_degree_pieces_config()
            ]

        else:
            raise NotImplementedError(gas)

    return out


def get_ch4_monthly_fifteen_degree_pieces_config() -> (
    CalculateCH4MonthlyFifteenDegreePieces
):
    interim_dir = "data/interim/ch4"

    return CalculateCH4MonthlyFifteenDegreePieces(
        step_config_id="only",
        gas="ch4",
        processed_bin_averages_file=f"{interim_dir}/ch4_observational-network_bin-averages.csv",
        observational_network_interpolated_file=f"{interim_dir}/ch4_observational-network_interpolated.nc",
        observational_network_global_annual_mean_file=f"{interim_dir}/ch4_observational-network_global-annual-mean.nc",
        lat_gradient_n_eofs_to_use="2",
        observational_network_latitudinal_gradient_eofs_file=f"{interim_dir}/ch4_observational-network_latitudinal-gradient-eofs.nc",
        observational_network_seasonality_file=f"{interim_dir}/ch4_observational-network_seasonality.nc",
        latitudinal_gradient_allyears_pcs_eofs_file=f"{interim_dir}/ch4_allyears-lat-gradient-eofs-pcs.nc",
        latitudinal_gradient_pc0_ch4_fossil_emissions_regression_file=f"{interim_dir}/ch4_pc0-ch4-fossil-emissions-regression.yaml",
        global_annual_mean_allyears_file=f"{interim_dir}/ch4_global-annual-mean_allyears.nc",
        global_annual_mean_allyears_monthly_file=f"{interim_dir}/ch4_global-annual-mean_allyears-monthly.nc",
        seasonality_allyears_fifteen_degree_monthly_file=f"{interim_dir}/ch4_seasonality_fifteen-degree_allyears-monthly.nc",
        latitudinal_gradient_fifteen_degree_allyears_monthly_file=f"{interim_dir}/ch4_latitudinal-gradient_fifteen-degree_allyears-monthly.nc",
    )
