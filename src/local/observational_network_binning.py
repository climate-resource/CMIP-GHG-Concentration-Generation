"""
Observational network binning functionality
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_handling import get_config_for_step_id

from local.config.base import Config
from local.dependencies import load_source_info_short_names


def get_obs_network_binning_input_files(  # noqa: PLR0911
    gas: str, config: Config, task_creation: bool = False
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network

    Parameters
    ----------
    gas
        Gas for which to retrieve the input files

    config
        Configuration instance to use for this retrieval

    task_creation
        Are we calling this function as part of config creation?

    Returns
    -------
    :
        Input files to use for binning the observational network
    """
    if gas in ("sf6",):
        return get_input_files_sf6_like(gas=gas, config=config, task_creation=task_creation)

    if gas in ("cfc11", "cfc12", "ch3ccl3", "ccl4"):
        return get_input_files_cfc11_like(gas=gas, config=config, task_creation=task_creation)

    if gas in ("cfc113",):
        return get_input_files_cfc113_like(gas=gas, config=config, task_creation=task_creation)

    if gas in ("cfc115", "cfc114", "hfc245fa", "c3f8", "cc4f8"):
        return get_input_files_cfc115_like(gas=gas, config=config, task_creation=task_creation)

    if gas in (
        "hfc134a",
        "ch2cl2",
        "ch3br",
        "ch3cl",
        "halon1211",
        "halon1301",
        "hcfc141b",
        "hcfc142b",
        "hcfc22",
        "hfc125",
        "hfc143a",
        "hfc152a",
        "hfc227ea",
        "hfc236fa",
        "hfc32",
        "hfc365mfc",
        "so2f2",
    ):
        return get_input_files_hfc134a_like(gas=gas, config=config, task_creation=task_creation)

    if gas in ("chcl3",):
        return get_input_files_chcl3_like(gas=gas, config=config, task_creation=task_creation)

    if gas in ("halon2402", "nf3", "c2f6", "cf4"):
        return get_input_files_halon2402_like(gas=gas, config=config, task_creation=task_creation)

    if gas in ("hfc23", "hfc4310mee"):
        return get_input_files_hfc23_like(gas=gas, config=config, task_creation=task_creation)

    raise NotImplementedError(gas)


def load_source_info_short_names_safe(fp: Path, task_creation: bool) -> list[str] | None:
    """
    Load source info short names, being safe with config creation
    """
    if task_creation:
        return None

    return load_source_info_short_names(fp)


def get_input_files_sf6_like(
    gas: str, config: Config, task_creation: bool
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network for gases we handle like SF6
    """
    # # Don't use SF6 surface flask to avoid double counting
    # config_process_noaa_surface_flask_data = get_config_for_step_id(
    #     config=config,
    #     step="process_noaa_surface_flask_data",
    #     step_config_id=gas,
    # )
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
        (
            config_process_noaa_hats_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_noaa_hats_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_md_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_md_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_medusa_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
    ]


def get_input_files_cfc11_like(
    gas: str, config: Config, task_creation: bool
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network for gases we handle like CFC-11
    """
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
    config_process_agage_gc_ms_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms_monthly",
    )
    config_process_agage_gc_ms_medusa_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms-medusa_monthly",
    )
    return [
        (
            config_process_noaa_hats_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_noaa_hats_data.source_info_short_names_file, task_creation=task_creation
            ),
        ),
        (
            config_process_agage_gc_md_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_md_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_medusa_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
    ]


def get_input_files_cfc113_like(
    gas: str, config: Config, task_creation: bool
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network for gases we handle like CFC-113
    """
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
        (
            config_process_noaa_hats_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_noaa_hats_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_md_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_md_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_medusa_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
    ]


def get_input_files_hfc134a_like(
    gas: str, config: Config, task_creation: bool
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network for gases we handle like HFC-134a
    """
    config_process_noaa_hats_data = get_config_for_step_id(
        config=config,
        step="process_noaa_hats_data",
        step_config_id=gas,
    )
    config_process_agage_gc_ms_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms_monthly",
    )
    config_process_agage_gc_ms_medusa_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms-medusa_monthly",
    )
    return [
        (
            config_process_noaa_hats_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_noaa_hats_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_medusa_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
    ]


def get_input_files_cfc115_like(
    gas: str, config: Config, task_creation: bool
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network for gases we handle like CFC-114
    """
    config_process_agage_gc_ms_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms_monthly",
    )
    config_process_agage_gc_ms_medusa_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms-medusa_monthly",
    )
    return [
        (
            config_process_agage_gc_ms_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_medusa_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
    ]


def get_input_files_chcl3_like(
    gas: str, config: Config, task_creation: bool
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network for gases we handle like CFC-11
    """
    config_process_agage_gc_md_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-md_monthly",
    )
    config_process_agage_gc_ms_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms_monthly",
    )
    config_process_agage_gc_ms_medusa_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms-medusa_monthly",
    )
    return [
        (
            config_process_agage_gc_md_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_md_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_medusa_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
    ]


def get_input_files_halon2402_like(
    gas: str, config: Config, task_creation: bool
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network for gases we handle like halon-2404
    """
    config_process_noaa_hats_data = get_config_for_step_id(
        config=config,
        step="process_noaa_hats_data",
        step_config_id=gas,
    )
    config_process_agage_gc_ms_medusa_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms-medusa_monthly",
    )
    return [
        (
            config_process_noaa_hats_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_noaa_hats_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
        (
            config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_medusa_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
    ]


def get_input_files_hfc23_like(
    gas: str, config: Config, task_creation: bool
) -> list[tuple[Path, list[str] | None]]:
    """
    Get the input files to use for binning the observational network for gases we handle like HFC-23
    """
    config_process_agage_gc_ms_medusa_data = get_config_for_step_id(
        config=config,
        step="retrieve_and_extract_agage_data",
        step_config_id=f"{gas}_gc-ms-medusa_monthly",
    )
    return [
        (
            config_process_agage_gc_ms_medusa_data.processed_monthly_data_with_loc_file,
            load_source_info_short_names_safe(
                config_process_agage_gc_ms_medusa_data.source_info_short_names_file,
                task_creation=task_creation,
            ),
        ),
    ]
