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
            known_hash="d46f08c9339ebea201abe99772505a903aa597c30c7a91372efd2cc063657d5a",
            url="https://doi.pangaea.de/10.1594/PANGAEA.899039?format=textfile",
        ),
    )
]
