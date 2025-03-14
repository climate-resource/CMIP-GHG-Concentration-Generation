"""
Global-mean extension handling
"""

from __future__ import annotations

from typing import Any

from pydoit_nb.config_handling import get_config_for_step_id

from local.config.base import Config


def get_global_mean_supplement_config(gas: str, config: Config) -> Any:
    """
    Get global-mean supplement config for a given gas

    Parameters
    ----------
    gas
        Gas

    config
        Configuration instance to use for this retrieval

    Returns
    -------
    :
        Global-mean supplement config, if one is used for this gas
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
        "ch3br",
        "ch3cl",
    ]:
        return get_config_for_step_id(
            config=config,
            step="retrieve_and_process_wmo_2022_ozone_assessment_ch7_data",
            step_config_id="only",
        )

    if gas in [
        "hcfc141b",
        "hcfc142b",
        "hcfc22",
    ]:
        return get_config_for_step_id(
            config=config,
            step="retrieve_and_process_western_et_al_2024_data",
            step_config_id="only",
        )

    if gas in [
        "hfc32",
        "hfc125",
        "hfc134a",
        "hfc143a",
        "hfc152a",
        "hfc227ea",
        "hfc236fa",
        "hfc245fa",
        "hfc365mfc",
        "hfc4310mee",
    ]:
        return get_config_for_step_id(
            config=config,
            step="retrieve_and_process_velders_et_al_2022_data",
            step_config_id="only",
        )

    if gas in [
        "hfc23",
    ]:
        return get_config_for_step_id(
            config=config,
            step="retrieve_and_process_adam_et_al_2024_data",
            step_config_id="only",
        )

    if gas in [
        "cf4",
        "c2f6",
        "c3f8",
    ]:
        return get_config_for_step_id(
            config=config,
            step="retrieve_and_process_trudinger_et_al_2016_data",
            step_config_id="only",
        )

    return None
