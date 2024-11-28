"""
Global-mean extension handling
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_handling import get_config_for_step_id

from local.config.base import Config


def get_global_mean_supplement_files(gas: str, config: Config) -> list[Path]:
    """
    Get global-mean supplement files for a given gas

    Parameters
    ----------
    gas
        Gas

    config
        Configuration instance to use for this retrieval

    Returns
    -------
    :
        Global-mean supplement files
    """
    if gas in [
        "cfc11",
        "cfc12",
        "cfc113",
        "cfc114",
        "cfc115",
        "ccl4",
        "ch3ccl3",
        "halon1211",
        "halon1301",
        "halon2402",
    ]:
        return [
            get_config_for_step_id(
                config=config,
                step="retrieve_and_process_wmo_2022_ozone_assessment_ch7_data",
                step_config_id="only",
            ).processed_data_file
        ]

    return []
