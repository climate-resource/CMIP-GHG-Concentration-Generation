"""
Creation of configuration for handling AGAGE's data
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_extract_agage import RetrieveExtractAGAGEDataConfig

DOWNLOAD_URLS = {
    ("ch4", "gc-md", "monthly"): [
        URLSource(
            known_hash="e6c3955c0e9178333c5f2177088a9fe84ec27b557901364750a82241f3477300",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_ch4_mon.txt",
        ),
        URLSource(
            known_hash="91cbef846e4158a880515b3b86b5b28d7510dcc6cf9494e3fec823e0c3f0678c",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_ch4_mon.txt",
        ),
        URLSource(
            known_hash="3d295bad0b883b6099ed5171044ed7a46e5ae93e8646a2020058a72c648ed0a6",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_ch4_mon.txt",
        ),
        URLSource(
            known_hash="e775e79fcf6cb833aa7d139c79725f25aefb81d4e90557616c4939d497f80719",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_ch4_mon.txt",
        ),
        URLSource(
            known_hash="fceb3a14534ce94d550f24831c7fc1258700f24b1a917005b6c06a85843ce0e1",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_ch4_mon.txt",
        ),
    ],
}


def create_agage_handling_config(
    data_sources: tuple[tuple[str, str, str]],
) -> list[RetrieveExtractAGAGEDataConfig]:
    """
    Create config for handling AGAGE data

    Parameters
    ----------
    data_sources
        Data sources to retrieve.
        Each input tuple should contain
        the gas of interest (zeroth element),
        the instrument of interest (first element)
        and the time frequency of interest (second element).

    Returns
    -------
        Configuration for handling AGAGE data for the requested data sources.
    """
    res = []
    for data_source in data_sources:
        gas, instrument, frequency = data_source

        raw_dir = Path("data/raw/agage/agage")
        interim_dir = Path("data/interim/agage/agage")

        res.append(
            RetrieveExtractAGAGEDataConfig(
                step_config_id=f"{gas}_{instrument}_{frequency}",
                gas=gas,
                instrument=instrument,
                time_frequency=frequency,
                raw_dir=raw_dir,
                download_complete_file=raw_dir
                / f"{gas}_{instrument}_{frequency}.complete",
                processed_monthly_data_with_loc_file=interim_dir
                / f"{gas}_{instrument}_{frequency}.csv",
                generate_hashes=False,
                download_urls=DOWNLOAD_URLS[(gas, instrument, frequency)],
            )
        )

    return res
