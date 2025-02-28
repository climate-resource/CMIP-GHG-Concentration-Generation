"""
Western et al. (2024) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_western_et_al_2024_data import (
    RetrieveExtractWesternEtal2024Data,
)
from local.dependencies import SourceInfo

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
        source_info=SourceInfo(
            short_name="Western et al., 2024",
            licence="CC BY 4.0",  # https://zenodo.org/records/10782689
            reference=(
                "Western, L.M., Daniel, J.S., Vollmer, M.K. et al. "
                "A decrease in radiative forcing "
                "and equivalent effective chlorine from hydrochlorofluorocarbons. "
                "Nat. Clim. Chang. 14, 805-807 (2024)."
            ),
            doi="https://doi.org/10.1038/s41558-024-02038-7",
            url="https://doi.org/10.1038/s41558-024-02038-7",
            resource_type="publication-article",
        ),
    )
]
