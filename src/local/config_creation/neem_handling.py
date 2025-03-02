"""
NEEM data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_process_neem_data import RetrieveProcessNEEMConfig
from local.dependencies import SourceInfo

RETRIEVE_AND_PROCESS_NEEM_STEPS = [
    RetrieveProcessNEEMConfig(
        step_config_id="only",
        raw_dir=Path("data/raw/neem"),
        processed_data_with_loc_file=Path("data/interim/neem/neem_with_location.csv"),
        download_url=URLSource(
            known_hash="d46f08c9339ebea201abe99772505a903aa597c30c7a91372efd2cc063657d5a",
            url="https://doi.pangaea.de/10.1594/PANGAEA.899039?format=textfile",
        ),
        source_info=SourceInfo(
            short_name="NEEM",
            licence="CC BY 4.0",
            reference=(
                "Rhodes, Rachael H; Brook, Edward J (2019): "
                "Methane in NEEM-2011-S1 ice core from North Greenland, "
                "1800 years continuous record: 5 year median, v2 [dataset]. "
                "PANGAEA, https://doi.org/10.1594/PANGAEA.899039, In supplement to: "
                "Rhodes, Rachael H; Faïn, Xavier; ...; Brook, Edward J (2013): "
                "Continuous methane measurements from a late Holocene Greenland ice core: "
                "Atmospheric and in-situ signals. Earth and Planetary Science Letters, 368, 9-19, "
                "https://doi.org/10.1016/j.epsl.2013.02.034"
            ),
            doi="https://doi.pangaea.de/10.1594/PANGAEA.899039",
            url="https://doi.pangaea.de/10.1594/PANGAEA.899039",
            resource_type="dataset",
        ),
    )
]
