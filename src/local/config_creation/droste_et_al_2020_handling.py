"""
Droste et al. (2020) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_droste_et_al_2020_data import (
    RetrieveExtractDrosteEtal2020Data,
)
from local.dependencies import SourceInfo

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
        source_info=SourceInfo(
            short_name="Droste et al., 2020",
            licence="CC BY 4.0",  # https://zenodo.org/records/3519317
            reference=(
                "Droste, E. S., Adcock, K. E., ..., Sturges, W. T., and Laube, J. C.: "
                "Trends and emissions of six perfluorocarbons "
                "in the Northern Hemisphere and Southern Hemisphere, "
                "Atmos. Chem. Phys., 20, 4787-4807, https://doi.org/10.5194/acp-20-4787-2020, 2020."
            ),
            doi="https://doi.org/10.5194/acp-20-4787-2020",
            url="https://doi.org/10.5194/acp-20-4787-2020",
            resource_type="publication-article",
        ),
    )
]
