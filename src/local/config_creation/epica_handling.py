"""
EPICA data handling config creation
"""

from __future__ import annotations

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_epica_data import RetrieveProcessEPICAConfig

RETRIEVE_AND_PROCESS_EPICA_STEPS = [
    RetrieveProcessEPICAConfig(
        step_config_id="only",
        raw_dir="data/raw/epica",
        processed_data_with_loc_file="data/interim/epica/epica_with_location.csv",
        download_url=URLSource(
            known_hash="26c9259d69bfe390f521d1f651de8ea37ece5bbb95b43df749ba4e00f763e9fd",
            url="https://doi.pangaea.de/10.1594/PANGAEA.552232?format=textfile",
        ),
    )
]
