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

IN_SITU_URL_BASE = "https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/in-situ/surface/{gas}_surface-insitu_ccgg_text.zip"
SURFACE_FLASK_URL_BASE = "https://gml.noaa.gov/aftp/data/trace_gases/{gas}/flask/surface/{gas}_surface-flask_ccgg_text.zip"


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
            res = f"https://gml.noaa.gov/aftp/data/hats/hcfcs/{gas.lower()}/flasks/{gas.upper()}_GCMS_flask.txt"

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
            res = f"https://gml.noaa.gov/aftp/data/hats/cfcs/{gas.lower()}/combined/HATS_global_{gas_hats}.txt"
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
            known_hash="92d8a6a6c59d936f1b338c0bf781009cd25348bf9a2c8dd9dde3cbf21e8dfe17",
        )
    ],
    ("co2", "in-situ"): [
        URLSource(
            url=IN_SITU_URL_BASE.format(gas="co2"),
            known_hash="0a68c9716bb9ec29e23966a2394e312618ed9b822885876d1ce5517bdf70acbe",
        )
    ],
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
            known_hash="c6067e98bf3896a45e21a248155bbf07815facce2c428bf015560602f31661f9",
        )
    ],
    ("cfc113", "hats"): [
        URLSource(
            url=get_hats_url("cfc113"),
            known_hash="7b7984976d6cadce14d27cfc67f38adba1686b6041c7676dc5296fca8ee9a3e0",
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
            known_hash="2537e02a6c4fc880c15db6ddf7ff0037add7e3f55fb227523e24ca16363128e0",
        )
    ],
    ("ch2cl2", "hats"): [
        URLSource(
            url=get_hats_url("ch2cl2"),
            known_hash="e32417315bbf068c52bb160f531785c74062e5055359f1f825939846bf4dc01c",
        )
    ],
    ("ch3br", "hats"): [
        URLSource(
            url=get_hats_url("ch3br"),
            known_hash="71d765146c9aec0a96d0ab5ba7a3640ea1ce734b669f2218adea1b6ecb8d093a",
        )
    ],
    ("ch3ccl3", "hats"): [
        URLSource(
            url=get_hats_url("ch3ccl3"),
            known_hash="8ea2b8e1317c6a62f409a52edf13bfecf7c960739fcbb7de0688688045b2f180",
        )
    ],
    ("ch3cl", "hats"): [
        URLSource(
            url=get_hats_url("ch3cl"),
            known_hash="8d6442dfc6216bb9ace91959ab49ad4bdbbef7042a2440d2d512660d22471080",
        )
    ],
    ("halon1211", "hats"): [
        URLSource(
            url=get_hats_url("halon1211"),
            known_hash="e09004be32c4a1cb34a55302e54b47dd1fd56a9724d562042d170d782bb7ee35",
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
            known_hash="5fdbbc85fcaf9d92ccec37e3f3f073df65d4f516fbe842d23342c13317205017",
        )
    ],
    ("hcfc141b", "hats"): [
        URLSource(
            url=get_hats_url("hcfc141b"),
            known_hash="a4a96582b90bb9bb2027b6db2de99f3c1aa9f0acdd042437d00ae340f487ebae",
        )
    ],
    ("hcfc142b", "hats"): [
        URLSource(
            url=get_hats_url("hcfc142b"),
            known_hash="f0e0cc665d4a16040ad8038b5c61626f1de7a6eca80b7e0162ca792c105d91ed",
        )
    ],
    ("hcfc22", "hats"): [
        URLSource(
            url=get_hats_url("hcfc22"),
            known_hash="f7bf03518216bdbe5096fd95d0bed2baf895a7ef0c1d81644408f41e8215cf77",
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
            known_hash="b4d7c2b760d13e2fe9f720b063dfec2b00f6ece65094d4a2e970bd53280a55a5",
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
            known_hash="13dc702e71a4f661ff18df23fd6ddb0dfd630289ab23f0c4356cb93e3ff02556",
        )
    ],
    ("hfc227ea", "hats"): [
        URLSource(
            url=get_hats_url("hfc227ea"),
            known_hash="c5d4bfb4f413015a8fb125528ce55aca9a6a9d67cfb933fa2c5ad6ff760b0b54",
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
            known_hash="e774693a7f1b252e48eb512d0f210a959e7ebd57c6ab5b0699d9f28563ffc206",
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

    elif network == "hats":
        out["process_noaa_hats_data"] = ProcessNOAAHATSDataConfig(  # type: ignore
            **process_step_attrs  # type: ignore
        )

    else:
        raise NotImplementedError(network)

    return cast(NOAAHandlingPieces, out)
