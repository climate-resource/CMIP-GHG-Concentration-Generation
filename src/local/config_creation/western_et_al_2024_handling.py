"""
Western et al. (2024) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_western_et_al_2024_data import (
    RetrieveExtractWesternEtal2024Data,
)

RAW_DIR = Path("data/raw/western-et-al-2024")

RETRIEVE_AND_PROCESS_WESTERN_ET_AL_2024_DATA_STEPS = [
    RetrieveExtractWesternEtal2024Data(
        step_config_id="only",
        zenodo_record=URLSource(
            url="https://zenodo.org/records/10782689/files/Projections.zip?download=1",
            known_hash="10ffeebdcfd362186ce64abb1dc1710e3ebf4d6b41bf18faf2bb7ff45a82b2f7",
        ),
        raw_dir=RAW_DIR,
        download_complete_file=RAW_DIR / "download.complete",
        processed_data_file=Path("data/interim/western-et-al-2024/western_et_al_2024.csv"),
    )
]
