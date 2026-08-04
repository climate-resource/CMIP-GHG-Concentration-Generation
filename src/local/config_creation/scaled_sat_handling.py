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


def create_scaled_sat_handling_config(
    data_sources: tuple[tuple[str, str, str], ...],
) -> ScaledSatHandlingPieces:
    """
    Create configuration for handling scaled_sat data

    Parameters
    ----------
    data_sources
        Data sources for which to create handling configuration.
        The zeroth element of each tuple should be the gas,
        the first element should be the network,
        the second element should be the fit to use.

    Returns
    -------
        Handling configuration for each data source in ``data_sources``.
        If ``data_sources`` is empty, the returned pieces still contain
        an (empty) entry for ``process_scaled_sat_data``.
    """
    res = defaultdict(list)
    res.setdefault("process_scaled_sat_data", [])
    for data_source in data_sources:
        pieces = create_scaled_sat_data_source_handling_pieces(
            gas=data_source[0], network=data_source[1], fit=data_source[2]
        )

        for key, value in pieces.items():
            res[key].append(value)

    return cast(ScaledSatHandlingPieces, res)


def create_scaled_sat_data_source_handling_pieces(
    gas: str, network: str, fit: str
) -> ScaledSatHandlingPieces:
    """
    Create the handling pieces for a given satellite data source

    Parameters
    ----------
    gas
        Gas for which to create the handling pieces

    network
        Network for which to create the handling pieces

    fit
        Fit to use for the scaled satellite data
        (e.g. ``"LINEAR_FIT"``, ``"ML_FIT"``, ``"NONLINEAR_LAT_FIT"``).
        This is the suffix of the OBS4MIPs/C3S file name for ``gas``.

    Returns
    -------
        Created handling pieces
    """
    out = {}
    scaled_data_path = Path(
        "data/raw/scaled_sat/"
        f"200301_202312-C3S-L3_X{gas.upper()}-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.6_{fit}.nc"
    )
    interim_data_path = Path(f"data/interim/scaled_sat/monthly_{gas}_scaled_sat.csv")

    process_step_attrs = dict(
        step_config_id=gas, gas=gas, scaled_data_path=scaled_data_path, interim_data_path=interim_data_path
    )

    if network == "scaled-sat":
        out["process_scaled_sat_data"] = ProcessScaledSatDataConfig(**process_step_attrs)

    else:
        raise NotImplementedError(network)

    return cast(ScaledSatHandlingPieces, out)
