"""
Trudinger et al. (2016) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_trudinger_et_al_2016_data import (
    RetrieveExtractTrudingerEtal2016Data,
)
from local.dependencies import SourceInfo

RAW_DIR = Path("data/raw/trudinger-et-al-2016")

RETRIEVE_AND_PROCESS_TRUDINGER_ET_AL_2016_DATA_STEPS = [
    RetrieveExtractTrudingerEtal2016Data(
        step_config_id="only",
        supplement=URLSource(
            url="https://acp.copernicus.org/articles/16/11733/2016/acp-16-11733-2016-supplement.zip",
            known_hash="48b7dfdf8310fb0aca0d49de129d5c4a7d6edd064ad8bfa8d2360f41edfb799b",
        ),
        raw_dir=RAW_DIR,
        download_complete_file=RAW_DIR / "download.complete",
        processed_data_file=Path("data/interim/trudinger-et-al-2016/trudinger_et_al_2016.csv"),
        source_info=SourceInfo(
            short_name="Trudinger et al., 2016",
            licence="CC BY 3.0",
            reference=(
                "Trudinger, C. M., Fraser, P. J., Etheridge, D. M., ..., O'Doherty, S., and Battle, M.: "
                "Atmospheric abundance and global emissions of perfluorocarbons CF4, C2F6 and C3F8 "
                "since 1800 inferred from ice core, firn, air archive and in situ measurements, "
                "Atmos. Chem. Phys., 16, 11733-11754, https://doi.org/10.5194/acp-16-11733-2016, 2016."
            ),
            doi="https://doi.org/10.5194/acp-16-11733-2016",
            url="https://doi.org/10.5194/acp-16-11733-2016",
            resource_type="publication-article",
        ),
    )
]
