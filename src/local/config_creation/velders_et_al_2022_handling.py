"""
Velders et al. (2022) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_velders_et_al_2022_data import (
    RetrieveExtractVeldersEtal2022Data,
)
from local.dependencies import SourceInfo

RAW_DIR = Path("data/raw/velders-et-al-2022")

RETRIEVE_AND_PROCESS_VELDERS_ET_AL_2022_DATA_STEPS = [
    RetrieveExtractVeldersEtal2022Data(
        step_config_id="only",
        raw_data_file_tmp=RAW_DIR / "KGL2021_constrProdEmis_ObsAgage_2500_OECD-SSP5.dat",
        zenodo_record=URLSource(
            url="https://zenodo.org/records/6520707/files/veldersguus/HFC-scenarios-2022-v1.0.zip?download=1",
            known_hash="74fe066fac06b742ba4fec6ad3af52a595f81a2a1c69d53a8eaf9ca846b3a7cd",
        ),
        raw_dir=RAW_DIR,
        download_complete_file=RAW_DIR / "download.complete",
        processed_data_file=Path("data/interim/velders-et-al-2022/velders_et_al_2022.csv"),
        source_info=SourceInfo(
            short_name="Velders et al., 2022",
            licence="Other (Open)",  # https://zenodo.org/records/6520707
            reference=(
                "Velders, G. J. M., Daniel, J. S., ... Weiss, R. F., and Young, D.: "
                "Projections of hydrofluorocarbon (HFC) emissions "
                "and the resulting global warming based on recent trends in observed abundances "
                "and current policies, "
                "Atmos. Chem. Phys., 22, 6087-6101, "
                "https://doi.org/10.5194/acp-22-6087-2022, 2022."
            ),
            doi="https://doi.org/10.5194/acp-22-6087-2022",
            url="https://doi.org/10.5194/acp-22-6087-2022",
            resource_type="publication-article",
        ),
    )
]
