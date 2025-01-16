"""
EPICA data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_epica_data import RetrieveProcessEPICAConfig

RETRIEVE_AND_PROCESS_EPICA_STEPS = [
    RetrieveProcessEPICAConfig(
        step_config_id="only",
        raw_dir=Path("data/raw/epica"),
        processed_data_with_loc_file=Path("data/interim/epica/epica_with_location.csv"),
        download_url=URLSource(
            known_hash="8cf4efd10a93a3783985c49e4dd0eba2aa02475f7afe7fbd147f7aae3229a267",
            url="https://doi.pangaea.de/10.1594/PANGAEA.552232?format=textfile",
        ),
    )
]
