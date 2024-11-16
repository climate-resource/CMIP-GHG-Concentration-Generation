"""
Calculate CO2 monthly 15 degree pieces notebook steps
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydoit_nb.config_handling import get_config_for_step_id
from pydoit_nb.notebook import ConfiguredNotebook, UnconfiguredNotebook
from pydoit_nb.notebook_step import UnconfiguredNotebookBasedStep

if TYPE_CHECKING:
    from ..config.base import Config, ConfigBundle


def configure_notebooks(
    unconfigured_notebooks: Iterable[UnconfiguredNotebook],
    config_bundle: ConfigBundle,
    step_name: str,
    step_config_id: str,
) -> list[ConfiguredNotebook]:
    """
    Configure notebooks

    Parameters
    ----------
    unconfigured_notebooks
        Unconfigured notebooks

    config_bundle
        Configuration bundle from which to take configuration values

    step_name
        Name of the step

    step_config_id
        Step config ID to use when configuring the notebook

    Returns
    -------
        Configured notebooks
    """
    uc_nbs_dict = {nb.notebook_path: nb for nb in unconfigured_notebooks}

    config = config_bundle.config_hydrated

    config_step = get_config_for_step_id(config=config, step=step_name, step_config_id=step_config_id)

    config_process_noaa_surface_flask_data = get_config_for_step_id(
        config=config,
        step="process_noaa_surface_flask_data",
        step_config_id=config_step.gas,
    )
    config_process_noaa_in_situ_data = get_config_for_step_id(
        config=config,
        step="process_noaa_in_situ_data",
        step_config_id=config_step.gas,
    )

    config_smooth_law_dome_data = get_config_for_step_id(
        config=config, step="smooth_law_dome_data", step_config_id=config_step.gas
    )

    config_retrieve_misc = get_config_for_step_id(
        config=config, step="retrieve_misc_data", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("12yy_co2-monthly-15-degree") / "1200_co2_bin-observational-network"
            ],
            configuration=(),
            dependencies=(
                config_process_noaa_surface_flask_data.processed_monthly_data_with_loc_file,
                config_process_noaa_in_situ_data.processed_monthly_data_with_loc_file,
            ),
            targets=(config_step.processed_bin_averages_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("12yy_co2-monthly-15-degree") / "1201_co2_interpolate-observational-network"
            ],
            configuration=(),
            dependencies=(config_step.processed_bin_averages_file,),
            targets=(config_step.observational_network_interpolated_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("12yy_co2-monthly-15-degree")
                / "1202_co2_observational-network-global-mean-latitudinal-gradient-seasonality"
            ],
            configuration=(),
            dependencies=(config_step.observational_network_interpolated_file,),
            targets=(
                config_step.observational_network_global_annual_mean_file,
                config_step.observational_network_latitudinal_gradient_eofs_file,
                config_step.observational_network_seasonality_file,
                config_step.observational_network_seasonality_change_eofs_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("12yy_co2-monthly-15-degree") / "1203_co2_extend-lat-gradient-pcs"
            ],
            configuration=(),
            dependencies=(
                config_step.observational_network_global_annual_mean_file,
                config_step.observational_network_latitudinal_gradient_eofs_file,
                config_retrieve_misc.primap.raw_dir
                / config_retrieve_misc.primap.download_url.url.split("/")[-1],
            ),
            targets=(
                config_step.latitudinal_gradient_allyears_pcs_eofs_file,
                config_step.latitudinal_gradient_pc0_co2_fossil_emissions_regression_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("12yy_co2-monthly-15-degree") / "1204_co2_extend-global-annual-mean"
            ],
            configuration=(),
            dependencies=(
                config_step.observational_network_global_annual_mean_file,
                config_step.latitudinal_gradient_allyears_pcs_eofs_file,
                config_smooth_law_dome_data.smoothed_median_file,
            ),
            targets=(config_step.global_annual_mean_allyears_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("12yy_co2-monthly-15-degree") / "1205_co2_extend-seasonality-change-pcs"
            ],
            configuration=(),
            dependencies=(
                config_step.global_annual_mean_allyears_file,
                config_step.observational_network_seasonality_change_eofs_file,
                config_retrieve_misc.hadcrut5.raw_dir
                / config_retrieve_misc.hadcrut5.download_url.url.split("/")[-1],
            ),
            targets=(
                config_step.seasonality_change_allyears_pcs_eofs_file,
                config_step.seasonality_change_temperature_co2_conc_regression_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("12yy_co2-monthly-15-degree") / "1206_co2_create-pieces-for-gridding"
            ],
            configuration=(),
            dependencies=(
                config_step.global_annual_mean_allyears_file,
                config_step.observational_network_seasonality_file,
                config_step.seasonality_change_allyears_pcs_eofs_file,
                config_step.latitudinal_gradient_allyears_pcs_eofs_file,
            ),
            targets=(
                config_step.global_annual_mean_allyears_monthly_file,
                config_step.seasonality_allyears_fifteen_degree_monthly_file,
                config_step.latitudinal_gradient_fifteen_degree_allyears_monthly_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[Config, ConfigBundle] = UnconfiguredNotebookBasedStep(
    step_name="calculate_co2_monthly_fifteen_degree_pieces",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("12yy_co2-monthly-15-degree") / "1200_co2_bin-observational-network",
            raw_notebook_ext=".py",
            summary="CO2 pieces - Bin observational data",
            doc="Bin the observational data for CO2.",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("12yy_co2-monthly-15-degree") / "1201_co2_interpolate-observational-network",
            raw_notebook_ext=".py",
            summary="CO2 pieces - Interpolate observational network onto our 15 degree x 60 degree grid",
            doc="Interpolate the observational data for CO2",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("12yy_co2-monthly-15-degree")
            / "1202_co2_observational-network-global-mean-latitudinal-gradient-seasonality",
            raw_notebook_ext=".py",
            summary="CO2 pieces - Observational network pieces",
            doc="Calculate latitudinal gradient, seasonality and global-mean from the observational network",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("12yy_co2-monthly-15-degree") / "1203_co2_extend-lat-gradient-pcs",
            raw_notebook_ext=".py",
            summary="CO2 pieces - Extend latitudinal gradient PCs",
            doc=(
                "Extend the latitudinal gradient principal components (PCs) "
                "over the entire time period of interest"
            ),
        ),
        UnconfiguredNotebook(
            notebook_path=Path("12yy_co2-monthly-15-degree") / "1204_co2_extend-global-annual-mean",
            raw_notebook_ext=".py",
            summary="CO2 pieces - Extend global, annual-mean over the entire time period",
            doc=(
                "Extend the global, annual-mean over the entire time period of interest "
                "using ice core records and our latitudinal gradient"
            ),
        ),
        UnconfiguredNotebook(
            notebook_path=Path("12yy_co2-monthly-15-degree") / "1205_co2_extend-seasonality-change-pcs",
            raw_notebook_ext=".py",
            summary="CO2 pieces - Extend seasonality change PCs",
            doc=(
                "Extend the seasonality change principal components (PCs) "
                "over the entire time period of interest"
            ),
        ),
        UnconfiguredNotebook(
            notebook_path=Path("12yy_co2-monthly-15-degree") / "1206_co2_create-pieces-for-gridding",
            raw_notebook_ext=".py",
            summary="CO2 pieces - Finalise the pieces for gridding",
            doc="Finalise the pieces required for creating gridded files",
        ),
    ],
    configure_notebooks=configure_notebooks,
)
