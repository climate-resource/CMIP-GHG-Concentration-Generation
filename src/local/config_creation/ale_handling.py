"""
ALE data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_extract_ale import RetrieveExtractALEDataConfig
from local.dependencies import SourceInfo

RETRIEVE_AND_EXTRACT_ALE_STEPS = [
    RetrieveExtractALEDataConfig(
        step_config_id="monthly",
        raw_dir=Path("data/raw/agage/ale"),
        processed_monthly_data_with_loc_file=Path("data/interim/agage/ale/monthly.csv"),
        download_complete_file=Path("data/raw/agage/ale/ale_monthly.complete"),
        download_urls=[
            URLSource(
                known_hash="38872b27c595bfb8a1509155bd713d2f519dab5c12b51e43f9256f8fa20ca040",
                url="https://agage2.eas.gatech.edu/data_archive/ale/monthly/ADR-ale.mon",
            ),
            URLSource(
                known_hash="7fd99c7f6014b9422da8144ff832e5b9b9ef143aa3f11ee199008d86528795b9",
                url="https://agage2.eas.gatech.edu/data_archive/ale/monthly/CGO-ale.mon",
            ),
            URLSource(
                known_hash="1603e2401243fa73e41ac45a840c8b17a8d46cf8219aac4ac77d9824a48ce658",
                url="https://agage2.eas.gatech.edu/data_archive/ale/monthly/ORG-ale.mon",
            ),
            URLSource(
                known_hash="d6f3e73214817262950b29dd10abc260c44cc1aaf4e371b7245b804d118c7d57",
                url="https://agage2.eas.gatech.edu/data_archive/ale/monthly/RPB-ale.mon",
            ),
            URLSource(
                known_hash="f12ffc3e4f31f77e449d12f924bffa5597c5595d09a94e46f9e716398981c845",
                url="https://agage2.eas.gatech.edu/data_archive/ale/monthly/SMO-ale.mon",
            ),
        ],
        source_info=SourceInfo(
            short_name="AGAGE ALE",
            licence="Free for scientific use, offer co-authorship. See https://www-air.larc.nasa.gov/missions/agage/data/policy",
            reference=(
                "Prinn et al., A history of chemically and radiatively important "
                "gases in air deduced from ALE/GAGE/AGAGE, J. Geophys. Res., 105, "
                "No. D14, p17,751-17,792, 2000."
            ),
            url="https://agage2.eas.gatech.edu/data_archive/ale/readme.ale",
            resource_type="dataset",
        ),
    )
]
