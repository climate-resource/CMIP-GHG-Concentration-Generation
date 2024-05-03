"""
NEEM data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_neem_data import RetrieveProcessNEEMConfig

RETRIEVE_AND_PROCESS_NEEM_STEPS = [
    RetrieveProcessNEEMConfig(
        step_config_id="only",
        raw_dir=Path("data/raw/neem"),
        processed_data_with_loc_file=Path("data/interim/neem/neem_with_location.csv"),
        download_url=URLSource(
            known_hash="3b57ca16db32f729a414422347f9292f2083c8d602f1f13d47a7fe7709d63d2d",
            url="https://doi.pangaea.de/10.1594/PANGAEA.899039?format=textfile",
        ),
    )
]
