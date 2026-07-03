"""
Creation of configuration for handling scaled sat data from OBS4MIPs
"""

from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

from local.config.process_scaled_sat_data import ProcessScaledSatDataConfig


class ScaledSatHandlingPieces(TypedDict):
    """NOAA handling pieces configuration"""

    process_scaled_sat_data: list[ProcessScaledSatDataConfig]
    """Configuration steps for processing the scaled sat data"""


def create_scaled_sat_handling_config(data_sources: tuple[tuple[str, str]]) -> ScaledSatHandlingPieces:
    """
    Create configuration for handling scaled_sat data

    Parameters
    ----------
    data_sources
        Data sources from NOAA for which to create handling configuration.
        The zeroth element of each tuple should be the gas,
        the first element should be the NOAA network.

    Returns
    -------
        Handling configuration for each data source in ``data_sources``
    """
    res = defaultdict(list)
    for data_source in data_sources:
        pieces = create_scaled_sat_data_source_handling_pieces(gas=data_source[0], network=data_source[1])

        for key, value in pieces.items():
            res[key].append(value)

    return cast(ScaledSatHandlingPieces, res)


def create_scaled_sat_data_source_handling_pieces(gas: str, network: str) -> ScaledSatHandlingPieces:
    """
    Create the handling pieces for a given NOAA data source

    Parameters
    ----------
    gas
        Gas for which to create the handling pieces

    network
        Network for which to create the handling pieces

    Returns
    -------
        Created handling pieces
    """
    out = {}
    scaled_data_path = Path(f"/home/anna_lanteri/data/scaled/{gas}_OBS4MIPs_to_all_gb_flask_linear_fit.nc")

    process_step_attrs = dict(step_config_id=gas, gas=gas, scaled_data_path=scaled_data_path)
    print("Network: " + network)

    if network == "scaled-sat":
        out["process_scaled_sat_data"] = ProcessScaledSatDataConfig(**process_step_attrs)
        print("")

    else:
        raise NotImplementedError(network)

    return cast(ScaledSatHandlingPieces, out)
