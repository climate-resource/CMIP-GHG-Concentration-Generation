"""
Creation of configuration for handling NOAA's data
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

from pydoit_nb.config_tools import URLSource

from local.config.process_noaa_hats_data import ProcessNOAAHATSDataConfig
from local.config.process_noaa_in_situ_data import ProcessNOAAInSituDataConfig
from local.config.process_noaa_surface_flask_data import (
    ProcessNOAASurfaceFlaskDataConfig,
)
from local.config.retrieve_and_extract_noaa import RetrieveExtractNOAADataConfig
from local.noaa_processing import HATS_GAS_NAME_MAPPING

IN_SITU_URL_BASE = (
    "https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/in-situ/surface/{gas}_surface-insitu_ccgg_text.zip"
)
SURFACE_FLASK_URL_BASE = (
    "https://gml.noaa.gov/aftp/data/trace_gases/{gas}/flask/surface/{gas}_surface-flask_ccgg_text.zip"
)


def get_hats_url(gas: str) -> str:  # noqa: PLR0912, PLR0915
    """
    Get URL for downloading from NOAA HATs

    Parameters
    ----------
    gas
        Gas for which to get the URL

    Returns
    -------
        URL from which to download the combined data
    """
    if "hcfc" in gas:
        if gas in ("hcfc142b", "hcfc22"):
            res = (
                f"https://gml.noaa.gov/aftp/data/hats/hcfcs/{gas.lower()}/flasks/{gas.upper()}_GCMS_flask.txt"
            )

        else:
            res = f"https://gml.noaa.gov/aftp/data/hats/hcfcs/{gas.lower()}/{gas.upper()}_GCMS_flask.txt"

    elif "cfc" in gas or "hfc" in gas:
        if gas in ("hfc125",):
            gas_hats = gas.replace("hfc", "hfc-")
        elif gas in ("hfc143a", "hfc227ea", "hfc236fa", "hfc32", "hfc365mfc"):
            gas_hats = gas.replace("hfc", "HFC-")
        elif gas in HATS_GAS_NAME_MAPPING:
            gas_hats = HATS_GAS_NAME_MAPPING[gas]
        else:
            gas_hats = gas

        if "cfc" in gas:
            res = (
                f"https://gml.noaa.gov/aftp/data/hats/cfcs/{gas.lower()}/combined/HATS_global_{gas_hats}.txt"
            )
        elif gas == "hfc152a":
            # Typo fun :)
            res = "https://gml.noaa.gov/aftp/data/hats/hfcs/hf152a_GCMS_flask.txt"
        elif gas in ("hfc125",):
            res = f"https://gml.noaa.gov/aftp/data/hats/hfcs/{gas_hats.upper()}_M2&PR1_MS_flask.txt"
        elif gas in ("hfc143a", "hfc32"):
            res = f"https://gml.noaa.gov/aftp/data/hats/hfcs/{gas_hats}_M2&PR1_MS_flask.txt"
        elif gas in ("hfc236fa",):
            res = f"https://gml.noaa.gov/aftp/data/hats/PERSEUS/{gas_hats}_PR1_MS_flask.txt"
        elif gas in ("hfc227ea", "hfc365mfc"):
            res = f"https://gml.noaa.gov/aftp/data/hats/hfcs/{gas_hats}_GCMS_flask.txt"
        elif "hfc" in gas:
            res = f"https://gml.noaa.gov/aftp/data/hats/hfcs/{gas_hats.lower()}_GCMS_flask.txt"
        else:
            raise NotImplementedError(gas)

    elif "halon" in gas:
        if gas in HATS_GAS_NAME_MAPPING:
            gas_hats = HATS_GAS_NAME_MAPPING[gas]
        else:
            gas_hats = gas

        if gas == "halon2402":
            res = "https://gml.noaa.gov/aftp/data/hats/halons/flasks/Hal2402_GCMS_flask.txt"

        elif gas == "halon1301":
            res = "https://gml.noaa.gov/aftp/data/hats/halons/flasks/H-1301_M2&PR1_MS_flask.txt"

        else:
            res = f"https://gml.noaa.gov/aftp/data/hats/halons/flasks/{gas_hats.upper()}_GCMS_flask.txt"

    elif gas in ("ch2cl2",):
        if gas in HATS_GAS_NAME_MAPPING:
            gas_hats = HATS_GAS_NAME_MAPPING[gas]
        else:
            gas_hats = gas

        res = f"https://gml.noaa.gov/aftp/data/hats/solvents/{gas_hats}/flasks/{gas.lower()}_GCMS_flask.txt"

    elif gas in ("ch3ccl3",):
        if gas in HATS_GAS_NAME_MAPPING:
            gas_hats = HATS_GAS_NAME_MAPPING[gas]
        else:
            gas_hats = gas

        res = f"https://gml.noaa.gov/aftp/data/hats/solvents/{gas_hats}/flasks/GCMS/{gas.upper()}_GCMS_flask.txt"

    elif gas in ("ccl4",):
        if gas in HATS_GAS_NAME_MAPPING:
            gas_hats = HATS_GAS_NAME_MAPPING[gas]
        else:
            gas_hats = gas

        res = f"https://gml.noaa.gov/aftp/data/hats/solvents/{gas_hats}/combined/HATS_global_{gas_hats}.txt"

    elif gas in ("ch3br",):
        res = f"https://gml.noaa.gov/aftp/data/hats/methylhalides/{gas}/flasks/{gas.upper()}_GCMS_flask.txt"

    elif gas in ("ch3cl",):
        if gas in HATS_GAS_NAME_MAPPING:
            gas_hats = HATS_GAS_NAME_MAPPING[gas]
        else:
            gas_hats = gas

        res = f"https://gml.noaa.gov/aftp/data/hats/methylhalides/{gas}/flasks/{gas_hats}_GCMS_flask.txt"

    elif gas in ("c2f6", "cf4"):
        if gas in HATS_GAS_NAME_MAPPING:
            # This is now a complete mess, anyway, can fix and refactor later
            gas_hats = HATS_GAS_NAME_MAPPING[gas].replace("_C", "")
        else:
            gas_hats = gas

        res = f"https://gml.noaa.gov/aftp/data/hats/PERSEUS/{gas_hats}_PR1_MS_flask.txt"

    elif gas in (
        "nf3",
        "so2f2",
    ):
        res = f"https://gml.noaa.gov/aftp/data/hats/PERSEUS/{gas.upper()}_PR1_MS_flask.txt"

    else:
        res = f"https://gml.noaa.gov/aftp/data/hats/{gas.lower()}/combined/GML_global_{gas.upper()}.txt"

    return res


DOWNLOAD_URLS = {
    ("co2", "surface-flask"): [
        URLSource(
            url=SURFACE_FLASK_URL_BASE.format(gas="co2"),
            known_hash="7e53e3aa7f43ec8300a080d9ae06dd64aa3cee526092e80e3a3d4989600f52ec",
        )
    ],
    ("co2", "in-situ"): [
        URLSource(
            url=IN_SITU_URL_BASE.format(gas="co2"),
            known_hash="3653b8b66e504da98edf256af23f827a0bf8e5e9a1422d979b38e65ad61499ae",
        )
    ],
    ("ch4", "in-situ"): [
        URLSource(
            url=IN_SITU_URL_BASE.format(gas="ch4"),
            known_hash="9eb2e865f79e6ca2fc5cd0963a52c90e0a8fd2f346b5d28d5e9989bd519d60cc",
        )
    ],
    ("ch4", "surface-flask"): [
        URLSource(
            url=SURFACE_FLASK_URL_BASE.format(gas="ch4"),
            known_hash="ea5fc01c59a67d1349ef2fdffd21ceb14ff0fb0d8518f1517d3972cac71df0e5",
        )
    ],
    ("n2o", "hats"): [
        URLSource(
            url=get_hats_url("n2o"),
            known_hash="d05fb01d87185d5020ca35a30ae40cc9c70fcc7d1e9d0640e43f09df9e568f1a",
        )
    ],
    ("n2o", "surface-flask"): [
        URLSource(
            url=SURFACE_FLASK_URL_BASE.format(gas="n2o"),
            known_hash="6b7e09c37b7fa456ab170a4c7b825b3d4b9f6eafb0ff61a2a46554b0e63e84b1",
        )
    ],
    ("c2f6", "hats"): [
        URLSource(
            url=get_hats_url("c2f6"),
            known_hash="648cc1699f3ef50a92ed8207bca8b1f3e14298d9653f598536dc80a9bbebf20c",
        )
    ],
    ("ccl4", "hats"): [
        URLSource(
            url=get_hats_url("ccl4"),
            known_hash="412139fa494c9cf43f6989403c37603168d57a2d0a7936a0cbc94f1599164310",
        )
    ],
    ("cf4", "hats"): [
        URLSource(
            url=get_hats_url("cf4"),
            known_hash="66955d644782c4148a7c367a6f02b96b92f060db7e5d0eb659bdfb6a3b8d1e35",
        )
    ],
    ("cfc11", "hats"): [
        URLSource(
            url=get_hats_url("cfc11"),
            known_hash="24412046f661bfe5b2463141ada01c114c353b19db01d8e452ec538f9596fcec",
        )
    ],
    ("cfc113", "hats"): [
        URLSource(
            url=get_hats_url("cfc113"),
            known_hash="e2b8369b85286203fcc61ae1cfddb5efb426013055a6b9163fe8db97ca619972",
        )
    ],
    ("cfc114", "hats"): [
        URLSource(
            url=get_hats_url("cfc114"),
            known_hash="7b7984976d6cadce14d27cfc67f38adba1686b6041c7676dc5296fca8ee9a3e0",
        )
    ],
    ("cfc12", "hats"): [
        URLSource(
            url=get_hats_url("cfc12"),
            known_hash="f3fe61be2b5f307da795568931d84399a34ecdde795d64ce1eeb8720df97bd89",
        )
    ],
    ("ch2cl2", "hats"): [
        URLSource(
            url=get_hats_url("ch2cl2"),
            known_hash="113ffef7b90c89ee9892cd1f3f441b6cb5a5d4c6330b21ad373eec38c0547816",
        )
    ],
    ("ch3br", "hats"): [
        URLSource(
            url=get_hats_url("ch3br"),
            known_hash="b4632358ecd69680cfd06ec248fe1e80fcfb3156626c73edc3d3c345ee1985e5",
        )
    ],
    ("ch3ccl3", "hats"): [
        URLSource(
            url=get_hats_url("ch3ccl3"),
            known_hash="6f5abb521a1bdfe2be264699270346a4f9b243db2cfba3d0b7af01d7bfc78ed3",
        )
    ],
    ("ch3cl", "hats"): [
        URLSource(
            url=get_hats_url("ch3cl"),
            known_hash="4c3bb2fc85cba33849b617727fd70d40611946338373c32dbda0ec515a2c2ac3",
        )
    ],
    ("halon1211", "hats"): [
        URLSource(
            url=get_hats_url("halon1211"),
            known_hash="b54b6f59477def77af961d1f212a2e41048e38199b4fa2a0b32c6ff9d31398e4",
        )
    ],
    ("halon1301", "hats"): [
        URLSource(
            url=get_hats_url("halon1301"),
            known_hash="92f70bff3d4009c0a6567df0de7ffb340d92ec1a6843c6d1483419f232f43669",
        )
    ],
    ("halon2402", "hats"): [
        URLSource(
            url=get_hats_url("halon2402"),
            known_hash="b94a7856bdcfaba8486992258b66187aeeb208389ded31bd8da006a185e5a4d7",
        )
    ],
    ("hcfc141b", "hats"): [
        URLSource(
            url=get_hats_url("hcfc141b"),
            known_hash="2a00218b2fb334f0fdd0941936a3142eaec7a578e1afc0a09021e1167dc7bb47",
        )
    ],
    ("hcfc142b", "hats"): [
        URLSource(
            url=get_hats_url("hcfc142b"),
            known_hash="bed61eb655cd316c8b1a5b23a9d9fe97f997077e1f1b40026531575fb28c4ebb",
        )
    ],
    ("hcfc22", "hats"): [
        URLSource(
            url=get_hats_url("hcfc22"),
            known_hash="1fc30f86323ca722865994ddc66d0e24e2dd581775e2f59267d4b1c57f7a9664",
        )
    ],
    ("hfc125", "hats"): [
        URLSource(
            url=get_hats_url("hfc125"),
            known_hash="d7ed3b857c8c7b483e1f26f4616ea049775f29956a652033ce75bb326b8424c9",
        )
    ],
    ("hfc134a", "hats"): [
        URLSource(
            url=get_hats_url("hfc134a"),
            known_hash="c8a9630b82aa9a936324ae10d1e79ed8f06aa382ce6d49d40aabbcecbb1249aa",
        )
    ],
    ("hfc143a", "hats"): [
        URLSource(
            url=get_hats_url("hfc143a"),
            known_hash="302f50d03b52b4e694ce44c8d7ddf7a7ae771e9f97cd649d4518578c55fe0580",
        )
    ],
    ("hfc152a", "hats"): [
        URLSource(
            url=get_hats_url("hfc152a"),
            known_hash="42aaa82a52fec1caa71656704357e0a3e64334c4fe3b6ac368abfeabaf6912bd",
        )
    ],
    ("hfc227ea", "hats"): [
        URLSource(
            url=get_hats_url("hfc227ea"),
            known_hash="cacda0e8941dcc456604f03da8a113d41763ab9c7c9f3f001a654d6b6d5be361",
        )
    ],
    ("hfc236fa", "hats"): [
        URLSource(
            url=get_hats_url("hfc236fa"),
            known_hash="93676d83d0ee49b0c3f8e7a372260839846f47e82e24a981d806d2c4d0211060",
        )
    ],
    ("hfc32", "hats"): [
        URLSource(
            url=get_hats_url("hfc32"),
            known_hash="dd3ec22a6c7d4c999b5fc9a5118efddcb4d5426527cebbe38d7d0cd076edefdb",
        )
    ],
    ("hfc365mfc", "hats"): [
        URLSource(
            url=get_hats_url("hfc365mfc"),
            known_hash="d8b98bc0420d8f93cab46b198cd40c716969d45251d0e20cc3c9574100a0429e",
        )
    ],
    ("nf3", "hats"): [
        URLSource(
            url=get_hats_url("nf3"),
            known_hash="55ad14efde473ff7c9529451cc2b9e4263ccf1c27fea20a1039b82e5da0ab309",
        )
    ],
    ("sf6", "hats"): [
        URLSource(
            url=get_hats_url("sf6"),
            known_hash="822543e2558e9e22e943478d37dffe0c758091c35d1ff9bf2b2697507dd3b39d",
        )
    ],
    ("sf6", "surface-flask"): [
        URLSource(
            url=SURFACE_FLASK_URL_BASE.format(gas="sf6"),
            known_hash="376c78456bba6844cca78ecd812b896eb2f10cc6b8a9bf6cad7a52dc39e31e9a",
        )
    ],
    ("so2f2", "hats"): [
        URLSource(
            url=get_hats_url("so2f2"),
            known_hash="1566e131da350ccad74bad652bf071c544c97787f9f9db0c3ee5f099e80275c8",
        )
    ],
}


class NOAAHandlingPieces(TypedDict):
    """NOAA handling pieces configuration"""

    retrieve_and_extract_noaa_data: list[RetrieveExtractNOAADataConfig]
    """Configuration steps for retrieving the NOAA data"""

    process_noaa_surface_flask_data: list[ProcessNOAASurfaceFlaskDataConfig]
    """Configuration steps for processing the NOAA surface flask data"""

    process_noaa_in_situ_data: list[ProcessNOAAInSituDataConfig]
    """Configuration steps for processing the NOAA in-situ data"""


def create_noaa_handling_config(data_sources: tuple[tuple[str, str]]) -> NOAAHandlingPieces:
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
        pieces = create_noaa_data_source_handling_pieces(gas=data_source[0], network=data_source[1])

        for key, value in pieces.items():
            res[key].append(value)

    return cast(NOAAHandlingPieces, res)


def create_noaa_data_source_handling_pieces(gas: str, network: str) -> NOAAHandlingPieces:
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
        source_info=interim_dir / f"source-info_{gas}_{network}.json",
    )
    if network == "surface-flask":
        interim_files["events_data"] = interim_dir / f"events_{gas}_{network}_raw-consolidated.csv"

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
        processed_monthly_data_with_loc_file=interim_dir / f"monthly_{gas}_{network}.csv",
        source_info_short_names_file=interim_dir / f"{gas}_{network}_source-short-names.txt",
    )
    if network == "surface-flask":
        out["process_noaa_surface_flask_data"] = ProcessNOAASurfaceFlaskDataConfig(  # type: ignore
            **process_step_attrs  # type: ignore
        )

    elif network == "in-situ":
        out["process_noaa_in_situ_data"] = ProcessNOAAInSituDataConfig(  # type: ignore
            **process_step_attrs  # type: ignore
        )

    elif network == "hats":
        out["process_noaa_hats_data"] = ProcessNOAAHATSDataConfig(  # type: ignore
            **process_step_attrs  # type: ignore
        )

    else:
        raise NotImplementedError(network)

    return cast(NOAAHandlingPieces, out)
