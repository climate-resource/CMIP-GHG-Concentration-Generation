"""
Ghosh et al. (2023) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from local.config.retrieve_and_process_ghosh_et_al_2023_data import (
    RetrieveExtractGhoshEtal2023Data,
)
from local.config.smooth_ghosh_et_al_2023_data import SmoothGhoshEtAl2023DataConfig

RAW_DIR = Path("data/raw/ghosh-et-al-2023")

RETRIEVE_AND_PROCESS_GHOSH_ET_AL_2023_DATA_STEPS = [
    RetrieveExtractGhoshEtal2023Data(
        step_config_id="only",
        raw_data_file=RAW_DIR / "Table_S2_N2O_data.xlsx",
        expected_hash="19dedb2f95732247c8a298ea0e38e8f2",
        processed_data_file=Path("data/interim/ghosh-et-al-2023/ghosh_et_al_2023.csv"),
    )
]
SMOOTH_GHOSH_ET_AL_2023_DATA_STEPS = [
    SmoothGhoshEtAl2023DataConfig(
        step_config_id="only",
        smoothed_file=Path("data/interim/ghosh_et_al_2023/smoothed.csv"),
    )
]
