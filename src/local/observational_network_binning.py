"""
Observational network binning functionality
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_handling import get_config_for_step_id

from local.config.base import Config


def get_obs_network_binning_input_files(gas: str, config: Config) -> list[Path]:
    """
    Get the input files to use for binning the observational network

    Parameters
    ----------
    gas
        Gas for which to retrieve the input files

    config
        Configuration instance to use for this retrieval

    Returns
    -------
        Input files to use for binning the observational network
    """
    if gas in ("sf6",):
        return get_input_files_sf6_like(gas=gas, config=config)

    raise NotImplementedError(gas)


def get_input_files_sf6_like(gas: str, config: Config) -> list[Path]:
    """
    Get the input files to use for binning the observational network for gases we handle like SF6
    """
    config_process_noaa_surface_flask_data = get_config_for_step_id(
        config=config,
        step="process_noaa_surface_flask_data",
        step_config_id=gas,
    )
    config_process_noaa_hats_data = get_config_for_step_id(
        config=config,
        step="process_noaa_hats_data",
        step_config_id=gas,
    )
    config_process_agage_gc_md_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-md_monthly",
    )
    config_process_agage_gc_ms_medusa_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms-medusa_monthly",
    )
    return [
        config_process_noaa_surface_flask_data.processed_monthly_data_with_loc_file,
        config_process_noaa_hats_data.processed_monthly_data_with_loc_file,
        config_process_agage_gc_md_data.processed_monthly_data_with_loc_file,
        config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
    ]
