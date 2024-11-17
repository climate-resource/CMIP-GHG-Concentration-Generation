"""
Calculate CH4 monthly 15 degree pieces notebook steps
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
    config_process_agage_data_gc_md = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{config_step.gas}_gc-md_monthly",
    )
    config_process_ale_data = get_config_for_step_id(
        config=config, step="retrieve_and_extract_ale_data", step_config_id="monthly"
    )
    config_process_gage_data = get_config_for_step_id(
        config=config, step="retrieve_and_extract_gage_data", step_config_id="monthly"
    )

    config_smooth_law_dome_data = get_config_for_step_id(
        config=config, step="smooth_law_dome_data", step_config_id=config_step.gas
    )
    config_process_epica = get_config_for_step_id(
        config=config, step="retrieve_and_process_epica_data", step_config_id="only"
    )
    config_process_neem = get_config_for_step_id(
        config=config, step="retrieve_and_process_neem_data", step_config_id="only"
    )
    config_retrieve_misc = get_config_for_step_id(
        config=config, step="retrieve_misc_data", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("11yy_ch4-monthly-15-degree") / "1100_ch4_bin-observational-network"
            ],
            configuration=(),
            dependencies=(
                config_process_noaa_surface_flask_data.processed_monthly_data_with_loc_file,
                config_process_noaa_in_situ_data.processed_monthly_data_with_loc_file,
                config_process_agage_data_gc_md.processed_monthly_data_with_loc_file,
                config_process_ale_data.processed_monthly_data_with_loc_file,
                config_process_gage_data.processed_monthly_data_with_loc_file,
            ),
            targets=(config_step.processed_bin_averages_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("11yy_ch4-monthly-15-degree") / "1101_ch4_interpolate-observational-network"
            ],
            configuration=(),
            dependencies=(config_step.processed_bin_averages_file,),
            targets=(config_step.observational_network_interpolated_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("11yy_ch4-monthly-15-degree") / "1102_ch4_global-mean-latitudinal-gradient-seasonality"
            ],
            configuration=(),
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
            unconfigured_notebook=uc_nbs_dict[Path("11yy_ch4-monthly-15-degree") / "1103_ch4_extend-pcs"],
            configuration=(),
            dependencies=(
                config_step.observational_network_global_annual_mean_file,
                config_step.observational_network_latitudinal_gradient_eofs_file,
                config_smooth_law_dome_data.smoothed_median_file,
                config_process_neem.processed_data_with_loc_file,
                config_retrieve_misc.primap.raw_dir
                / config_retrieve_misc.primap.download_url.url.split("/")[-1],
            ),
            targets=(
                config_step.latitudinal_gradient_allyears_pcs_eofs_file,
                config_step.latitudinal_gradient_pc0_ch4_fossil_emissions_regression_file,
            ),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("11yy_ch4-monthly-15-degree") / "1104_ch4_extend-global-annual-mean"
            ],
            configuration=(),
            dependencies=(
                config_step.observational_network_global_annual_mean_file,
                config_step.latitudinal_gradient_allyears_pcs_eofs_file,
                config_smooth_law_dome_data.smoothed_median_file,
                config_process_neem.processed_data_with_loc_file,
                config_process_epica.processed_data_with_loc_file,
            ),
            targets=(config_step.global_annual_mean_allyears_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        ),
        ConfiguredNotebook(
            unconfigured_notebook=uc_nbs_dict[
                Path("11yy_ch4-monthly-15-degree") / "1105_ch4_create-pieces-for-gridding"
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


step: UnconfiguredNotebookBasedStep[Config, ConfigBundle] = UnconfiguredNotebookBasedStep(
    step_name="calculate_ch4_monthly_fifteen_degree_pieces",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("11yy_ch4-monthly-15-degree") / "1100_ch4_bin-observational-network",
            raw_notebook_ext=".py",
            summary="CH4 pieces - Bin observational data",
            doc="Bin the observational data for CH4.",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("11yy_ch4-monthly-15-degree") / "1101_ch4_interpolate-observational-network",
            raw_notebook_ext=".py",
            summary="CH4 pieces - Interpolate observational network onto our 15 degree x 60 degree grid",
            doc="Interpolate the observational data for CH4.",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("11yy_ch4-monthly-15-degree")
            / "1102_ch4_global-mean-latitudinal-gradient-seasonality",
            raw_notebook_ext=".py",
            summary="CH4 pieces - Observational network pieces",
            doc="Calculate latitudinal gradient, seasonality and global-mean from the observational network",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("11yy_ch4-monthly-15-degree") / "1103_ch4_extend-pcs",
            raw_notebook_ext=".py",
            summary="CH4 pieces - Extend PCs over the entire time period",
            doc="Extend the principal components (PCs) over the entire time period of interest",
        ),
        UnconfiguredNotebook(
            notebook_path=Path("11yy_ch4-monthly-15-degree") / "1104_ch4_extend-global-annual-mean",
            raw_notebook_ext=".py",
            summary="CH4 pieces - Extend global, annual-mean over the entire time period",
            doc=(
                "Extend the global, annual-mean over the entire time period of interest "
                "using ice core records and our latitudinal gradient."
            ),
        ),
        UnconfiguredNotebook(
            notebook_path=Path("11yy_ch4-monthly-15-degree") / "1105_ch4_create-pieces-for-gridding",
            raw_notebook_ext=".py",
            summary="CH4 pieces - Finalise the pieces for gridding",
            doc="Finalise the pieces required for creating gridded files",
        ),
    ],
    configure_notebooks=configure_notebooks,
)
