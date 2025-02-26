"""
Droste et al. (2020) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_scripps_data import (
    RetrieveProcessScrippsConfig,
    ScrippsSource,
)

RETRIEVE_AND_PROCESS_SCRIPPS_DATA = [
    RetrieveProcessScrippsConfig(
        step_config_id="only",
        raw_dir=Path("data/raw/mauna_loa/"),
        merged_ice_core_data=URLSource(
            url="https://scrippsco2.ucsd.edu/assets/data/atmospheric/merged_ice_core_mlo_spo/spline_merged_ice_core_yearly.csv",
            known_hash="1d9a64819477180f48426e3b23b221b59801d13dfde594b23ee12784a470b340",
        ),
        merged_ice_core_data_processed_data_file=Path("data/interim/mauna_loa/merged_ice_core.csv"),
        station_data=[
            ScrippsSource(
                # TODO: rest of these (?)
                url_source=URLSource(
                    url="https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/merged_in_situ_and_flask/monthly/monthly_merge_co2_ptb.csv",
                    known_hash="0b75a1e45bbc4157037adab240cc4560e1bb149da256b9424ff94c7dacb9084a",
                ),
                station_code="ptb",
                lat="71.3 N",
                lon="156.6 W",
            )
        ],
        processed_data_with_loc_file=Path("data/interim/mauna_loa/stations_with_loc.csv"),
    )
]
