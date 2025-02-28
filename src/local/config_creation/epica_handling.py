"""
EPICA data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_epica_data import RetrieveProcessEPICAConfig
from local.dependencies import SourceInfo

RETRIEVE_AND_PROCESS_EPICA_STEPS = [
    RetrieveProcessEPICAConfig(
        step_config_id="only",
        raw_dir=Path("data/raw/epica"),
        processed_data_with_loc_file=Path("data/interim/epica/epica_with_location.csv"),
        download_url=URLSource(
            known_hash="8cf4efd10a93a3783985c49e4dd0eba2aa02475f7afe7fbd147f7aae3229a267",
            url="https://doi.pangaea.de/10.1594/PANGAEA.552232?format=textfile",
        ),
        source_info=SourceInfo(
            short_name="EPICA",
            licence="CC BY 3.0",
            reference=(
                "EPICA Community Members (2006): Methane of ice core EDML [dataset]. "
                "PANGAEA, https://doi.org/10.1594/PANGAEA.552232, In supplement to: "
                "Barbante, Carlo; Barnola, Jean-Marc; ... Wolff, Eric William (2006): "
                "One-to-one coupling of glacial climate variability in Greenland and Antarctica. "
                "Nature, 444, 195-198, https://doi.org/10.1038/nature05301"
            ),
            doi="https://doi.pangaea.de/10.1594/PANGAEA.552232",
            url="https://doi.pangaea.de/10.1594/PANGAEA.552232",
            resource_type="dataset",
        ),
    )
]
