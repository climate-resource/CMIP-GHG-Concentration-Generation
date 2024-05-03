"""
Creation of configuration for handling NOAA's data
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

from pydoit_nb.config_tools import URLSource

from local.config.process_noaa_in_situ_data import ProcessNOAAInSituDataConfig
from local.config.process_noaa_surface_flask_data import (
    ProcessNOAASurfaceFlaskDataConfig,
)
from local.config.retrieve_and_extract_noaa import RetrieveExtractNOAADataConfig

IN_SITU_URL_BASE = "https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/in-situ/surface/{gas}_surface-insitu_ccgg_text.zip"
SURFACE_FLASK_URL_BASE = "https://gml.noaa.gov/aftp/data/trace_gases/{gas}/flask/surface/{gas}_surface-flask_ccgg_text.zip"

DOWNLOAD_URLS = {
    ("ch4", "in-situ"): [
        URLSource(
            url=IN_SITU_URL_BASE.format(gas="ch4"),
            known_hash="c8ad74288d860c63b6a027df4d7bf6742e772fc4e3f99a4052607a382d7fefb2",
        )
    ],
    ("ch4", "surface-flask"): [
        URLSource(
            url=SURFACE_FLASK_URL_BASE.format(gas="ch4"),
            known_hash="e541578315328857f01eb7432b5949e39beabab2017c09e46727ac49ec728087",
        )
    ],
}
#     (
#         "co2",
#         "surface-flask",
#         "92d8a6a6c59d936f1b338c0bf781009cd25348bf9a2c8dd9dde3cbf21e8dfe17",
#     ),
#     (
#         "co2",
#         "in-situ",
#         "0a68c9716bb9ec29e23966a2394e312618ed9b822885876d1ce5517bdf70acbe",
#     ),
#     (
#         "ch4",
#         "surface-flask",
#         "e541578315328857f01eb7432b5949e39beabab2017c09e46727ac49ec728087",
#     ),
#     (
#         "ch4",
#         "in-situ",
#         "c8ad74288d860c63b6a027df4d7bf6742e772fc4e3f99a4052607a382d7fefb2",
#     ),
#     (
#         "n2o",
#         "surface-flask",
#         "6b7e09c37b7fa456ab170a4c7b825b3d4b9f6eafb0ff61a2a46554b0e63e84b1",
#     ),
#     (
#         "sf6",
#         "surface-flask",
#         "376c78456bba6844cca78ecd812b896eb2f10cc6b8a9bf6cad7a52dc39e31e9a",
#     ),
# )


class NOAAHandlingPieces(TypedDict):
    """NOAA handling pieces configuration"""

    retrieve_and_extract_noaa_data: list[RetrieveExtractNOAADataConfig]
    """Configuration steps for retrieving the NOAA data"""

    process_noaa_surface_flask_data: list[ProcessNOAASurfaceFlaskDataConfig]
    """Configuration steps for processing the NOAA surface flask data"""

    process_noaa_in_situ_data: list[ProcessNOAAInSituDataConfig]
    """Configuration steps for processing the NOAA in-situ data"""


def create_noaa_handling_config(
    data_sources: tuple[tuple[str, str]]
) -> NOAAHandlingPieces:
    """
    Create configuration for handling NOAA data

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
        pieces = create_noaa_data_source_handling_pieces(
            gas=data_source[0], network=data_source[1]
        )

        for key, value in pieces.items():
            res[key].append(value)

    return cast(NOAAHandlingPieces, res)


def create_noaa_data_source_handling_pieces(
    gas: str, network: str
) -> NOAAHandlingPieces:
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

    raw_dir = Path("data/raw/noaa")
    interim_dir = Path("data/interim/noaa")
    interim_files = dict(
        monthly_data=interim_dir / f"monthly_{gas}_{network}_raw-consolidated.csv",
    )
    if network == "surface-flask":
        interim_files["events_data"] = (
            interim_dir / f"events_{gas}_{network}_raw-consolidated.csv"
        )

    out["retrieve_and_extract_noaa_data"] = RetrieveExtractNOAADataConfig(
        step_config_id=f"{gas}_{network}",
        gas=gas,
        source=network,
        raw_dir=raw_dir,
        download_complete_file=raw_dir / f"{gas}_{network}.complete",
        interim_files=interim_files,
        download_urls=DOWNLOAD_URLS[(gas, network)],
    )

    process_step_attrs = dict(
        step_config_id=gas,
        gas=gas,
        processed_monthly_data_with_loc_file=interim_dir
        / f"monthly_{gas}_{network}.csv",
    )
    if network == "surface-flask":
        out["process_noaa_surface_flask_data"] = ProcessNOAASurfaceFlaskDataConfig(  # type: ignore
            **process_step_attrs  # type: ignore
        )

    elif network == "in-situ":
        out["process_noaa_in_situ_data"] = ProcessNOAAInSituDataConfig(  # type: ignore
            **process_step_attrs  # type: ignore
        )

    else:
        raise NotImplementedError(network)

    return cast(NOAAHandlingPieces, out)
