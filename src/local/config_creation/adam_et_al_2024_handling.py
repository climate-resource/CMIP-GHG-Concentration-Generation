"""
Adam et al. (2024) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from local.config.retrieve_and_process_adam_et_al_2024_data import (
    RetrieveExtractAdamEtal2024Data,
)
from local.dependencies import SourceInfo

RAW_DIR = Path("data/raw/adam-et-al-2024")

RETRIEVE_AND_PROCESS_ADAM_ET_AL_2024_DATA_STEPS = [
    RetrieveExtractAdamEtal2024Data(
        step_config_id="only",
        raw_data_file=RAW_DIR / "HFC-23_Global_annual_mole_fraction.csv",
        processed_data_file=Path("data/interim/adam-et-al-2024/adam_et_al_2024.csv"),
        source_info=SourceInfo(
            short_name="Adam et al., 2024",
            # Effectively AGAGE data
            licence="Free for scientific use, offer co-authorship. See https://www-air.larc.nasa.gov/missions/agage/data/policy",
            reference=(
                "Adam, B., Western, L.M., Mühle, J. et al. "
                "Emissions of HFC-23 do not reflect commitments made under the Kigali Amendment. "
                "Commun Earth Environ 5, 783 (2024)."
            ),
            doi="https://doi.org/10.1038/s43247-024-01946-y",
            url="https://doi.org/10.1038/s43247-024-01946-y",
            resource_type="publication-article",
        ),
    )
]
