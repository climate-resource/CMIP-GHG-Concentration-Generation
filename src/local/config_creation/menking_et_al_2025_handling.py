"""
Menking et al. (2025) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from local.config.retrieve_and_process_menking_et_al_2025_data import (
    RetrieveExtractMenkingEtal2025Data,
)

RAW_DIR = Path("data/raw/menking-et-al-2025")

RETRIEVE_AND_PROCESS_MENKING_ET_AL_2025_DATA_STEPS = [
    RetrieveExtractMenkingEtal2025Data(
        step_config_id="only",
        raw_data_file=RAW_DIR / "spline-fits-for-ZN_CMIP7.xlsx",
        expected_hash="20df9337cddb739dc805e53847a19424",
        processed_data_file=Path("data/interim/menking-et-al-2025/menking_et_al_2025.csv"),
    )
]
