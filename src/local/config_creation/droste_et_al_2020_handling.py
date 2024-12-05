"""
Droste et al. (2020) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_droste_et_al_2020_data import (
    RetrieveExtractDrosteEtal2020Data,
)

RAW_DIR = Path("data/raw/droste-et-al-2020")

RETRIEVE_AND_PROCESS_DROSTE_ET_AL_2020_DATA_STEPS = [
    RetrieveExtractDrosteEtal2020Data(
        step_config_id="only",
        zenodo_record=URLSource(
            url="https://zenodo.org/records/3519317/files/Trends-Emission_PFCs_Droste-etal_ACP_20191025.zip?download=1",
            known_hash="f71fda6b8848f627b7736870241bfa075d941bcd458fff6105287e951aed6c21",
        ),
        raw_dir=RAW_DIR,
        download_complete_file=RAW_DIR / "download.complete",
        processed_data_file=Path("data/interim/droste-et-al-2020/droste_et_al_2020.csv"),
    )
]
