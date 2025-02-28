"""
GAGE data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_extract_gage import RetrieveExtractGAGEDataConfig
from local.dependencies import SourceInfo

RETRIEVE_AND_EXTRACT_GAGE_STEPS = [
    RetrieveExtractGAGEDataConfig(
        step_config_id="monthly",
        raw_dir=Path("data/raw/agage/gage"),
        processed_monthly_data_with_loc_file=Path("data/interim/agage/gage/monthly.csv"),
        download_complete_file=Path("data/raw/agage/gage/gage_monthly.complete"),
        download_urls=[
            URLSource(
                known_hash="3955484431eb728cfcbb42df2364939870f2b444367c9dab0c875052b6ff40ff",
                url="https://agage2.eas.gatech.edu/data_archive/gage/monthly/CGO-gage.mon",
            ),
            URLSource(
                known_hash="8e7fd65035cf5b79da473d59bda2b1f499b35d211003f64a9e268cab04c59ea7",
                url="https://agage2.eas.gatech.edu/data_archive/gage/monthly/MHD-gage.mon",
            ),
            URLSource(
                known_hash="31deafb97e07c390b5d0647f655a3dc9d82c74b918305f99505cdd6595c4ea99",
                url="https://agage2.eas.gatech.edu/data_archive/gage/monthly/ORG-gage.mon",
            ),
            URLSource(
                known_hash="b2608a3836ed41c925d7b30395fefced4e9f1706be620a607401d6f3ba578447",
                url="https://agage2.eas.gatech.edu/data_archive/gage/monthly/RPB-gage.mon",
            ),
            URLSource(
                known_hash="5705e7d9d0f57de33ec4baaa66f9ec78d5c053f50949f1e5f60d1e8d4af5a40b",
                url="https://agage2.eas.gatech.edu/data_archive/gage/monthly/SMO-gage.mon",
            ),
        ],
        source_info=SourceInfo(
            short_name="AGAGE GAGE",
            licence="Free for scientific use, offer co-authorship. See https://www-air.larc.nasa.gov/missions/agage/data/policy",
            reference=(
                "Prinn et al., A history of chemically and radiatively important "
                "gases in air deduced from ALE/GAGE/AGAGE, J. Geophys. Res., 105, "
                "No. D14, p17,751-17,792, 2000."
            ),
            url="https://agage2.eas.gatech.edu/data_archive/gage/readme.gage",
            resource_type="dataset",
        ),
    )
]
