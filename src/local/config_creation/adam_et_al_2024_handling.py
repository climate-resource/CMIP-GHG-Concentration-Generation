"""
Adam et al. (2024) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from local.config.retrieve_and_process_adam_et_al_2024_data import (
    RetrieveExtractAdamEtal2024Data,
)

RAW_DIR = Path("data/raw/adam-et-al-2024")

RETRIEVE_AND_PROCESS_ADAM_ET_AL_2024_DATA_STEPS = [
    RetrieveExtractAdamEtal2024Data(
        step_config_id="only",
        raw_data_file=RAW_DIR / "HFC-23_Global_annual_mole_fraction.csv",
        processed_data_file=Path("data/interim/adam-et-al-2024/adam_et_al_2024.csv"),
    )
]
