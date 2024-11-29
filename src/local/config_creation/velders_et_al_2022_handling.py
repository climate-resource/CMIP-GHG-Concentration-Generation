"""
Velders et al. (2022) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_velders_et_al_2022_data import (
    RetrieveExtractVeldersEtal2022Data,
)

RAW_DIR = Path("data/raw/velders-et-al-2022")

RETRIEVE_AND_PROCESS_VELDERS_ET_AL_2022_DATA_STEPS = [
    RetrieveExtractVeldersEtal2022Data(
        step_config_id="only",
        zenodo_record=URLSource(
            url="https://zenodo.org/records/6520707/files/veldersguus/HFC-scenarios-2022-v1.0.zip?download=1",
            known_hash="74fe066fac06b742ba4fec6ad3af52a595f81a2a1c69d53a8eaf9ca846b3a7cd",
        ),
        raw_dir=RAW_DIR,
        download_complete_file=RAW_DIR / "download.complete",
        processed_data_file=Path("data/interim/velders-et-al-2022/velders_et_al_2022.csv"),
    )
]
