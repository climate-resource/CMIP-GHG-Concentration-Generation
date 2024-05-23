"""
Calculate SF6-like gases monthly 15 degree pieces notebook steps
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydoit_nb.config_handling import get_config_for_step_id
from pydoit_nb.notebook import ConfiguredNotebook, UnconfiguredNotebook
from pydoit_nb.notebook_step import UnconfiguredNotebookBasedStep

from local.observational_network_binning import get_obs_network_binning_input_files

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

    config_step = get_config_for_step_id(
        config=config, step=step_name, step_config_id=step_config_id
    )

    obs_network_input_files = get_obs_network_binning_input_files(
        gas=config_step.gas, config=config
    )

    config_historical_emissions = get_config_for_step_id(
        config=config, step="compile_historical_emissions", step_config_id="only"
    )
    # config_retrieve_misc = get_config_for_step_id(
    #     config=config, step="retrieve_misc_data", step_config_id="only"
    # )

    global_mean_supplement_files = get_obs_network_binning_input_files(
        gas=config_step.gas, config=config
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("13yy_sf6-like-monthly-15-degree")
                / "1300_sf6-like_bin-observational-network"
            ],
            configuration=(),
            dependencies=tuple(obs_network_input_files),
            targets=(config_step.processed_bin_averages_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("13yy_sf6-like-monthly-15-degree")
                / "1301_sf6-like_interpolate-observational-network"
            ],
            configuration=(
                config_step.allow_poleward_extension,
                config_step.allow_long_poleward_extension,
            ),
            dependencies=(config_step.processed_bin_averages_file,),
            targets=(config_step.observational_network_interpolated_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("13yy_sf6-like-monthly-15-degree")
                / "1302_sf6-like_observational-network-global-mean-latitudinal-gradient-seasonality"
            ],
            configuration=(
                config_step.year_drop_observational_data_before_and_including,
                config_step.year_drop_observational_data_after_and_including,
            ),
            dependencies=(config_step.observational_network_interpolated_file,),
            targets=(
                config_step.observational_network_global_annual_mean_file,
                config_step.observational_network_latitudinal_gradient_eofs_file,
                config_step.observational_network_seasonality_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("13yy_sf6-like-monthly-15-degree")
                / "1303_sf6-like_extend-lat-gradient-pcs"
            ],
            configuration=(),
            dependencies=(
                config_step.observational_network_global_annual_mean_file,
                config_step.observational_network_latitudinal_gradient_eofs_file,
                config_historical_emissions.complete_historical_emissions_file,
            ),
            targets=(
                config_step.latitudinal_gradient_allyears_pcs_eofs_file,
                config_step.latitudinal_gradient_pc0_total_emissions_regression_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("13yy_sf6-like-monthly-15-degree")
                / "1304_sf6-like_extend-global-annual-mean"
            ],
            configuration=(),
            dependencies=(
                config_step.observational_network_global_annual_mean_file,
                config_step.latitudinal_gradient_allyears_pcs_eofs_file,
                *global_mean_supplement_files,
            ),
            targets=(config_step.global_annual_mean_allyears_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("13yy_sf6-like-monthly-15-degree")
                / "1305_sf6-like_create-pieces-for-gridding"
            ],
            configuration=(),
            dependencies=(
                config_step.global_annual_mean_allyears_file,
                config_step.observational_network_seasonality_file,
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


step: UnconfiguredNotebookBasedStep[
    Config, ConfigBundle
] = UnconfiguredNotebookBasedStep(
    step_name="calculate_sf6_like_monthly_fifteen_degree_pieces",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("13yy_sf6-like-monthly-15-degree")
            / "1300_sf6-like_bin-observational-network",
            raw_notebook_ext=".py",
            summary="SF6-like gas pieces - Bin observational data",
            doc="Bin the observational data for SF6-like gases.",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("13yy_sf6-like-monthly-15-degree")
            / "1301_sf6-like_interpolate-observational-network",
            raw_notebook_ext=".py",
            summary="SF6-like gas pieces - Interpolate observational network onto our 15 deg x 60 deg grid",
            doc="Interpolate the observational data for SF6-like gases.",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("13yy_sf6-like-monthly-15-degree")
            / "1302_sf6-like_observational-network-global-mean-latitudinal-gradient-seasonality",
            raw_notebook_ext=".py",
            summary="SF6-like gas pieces - Observational network pieces",
            doc="Calculate latitudinal gradient, seasonality and global-mean from the observational network",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("13yy_sf6-like-monthly-15-degree")
            / "1303_sf6-like_extend-lat-gradient-pcs",
            raw_notebook_ext=".py",
            summary="SF6-like gas pieces - Extend PCs over the entire time period",
            doc="Extend the principal components (PCs) over the entire time period of interest",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("13yy_sf6-like-monthly-15-degree")
            / "1304_sf6-like_extend-global-annual-mean",
            raw_notebook_ext=".py",
            summary="SF6-like gas pieces - Extend global, annual-mean over the entire time period",
            doc=(
                "Extend the global, annual-mean over the entire time period of interest "
                "using other data sources and our latitudinal gradient."
            ),
        ),
        UnconfiguredNotebook(
            notebook_path=Path("13yy_sf6-like-monthly-15-degree")
            / "1305_sf6-like_create-pieces-for-gridding",
            raw_notebook_ext=".py",
            summary="SF6-like gas pieces - Finalise the pieces for gridding",
            doc="Finalise the pieces required for creating gridded files",
        ),
    ],
    configure_notebooks=configure_notebooks,
)
