"""
Config for retrieving, extracting and processing Trudinger et al. (2016) data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource


@frozen
class RetrieveExtractTrudingerEtal2016Data:
    """
    Config for retrieving, extracting and processing Trudinger et al. (2016) data

    Original paper: https://doi.org/10.5194/acp-16-11733-2016
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    supplement: URLSource
    """Supplement from which to download the raw data"""

    raw_dir: Path
    """
    File in which to save the raw data
    """

    download_complete_file: Path
    """
    Path in which to write the time at which the download was completed

    This is mainly used to help with setting the dependencies between notebooks correctly.
    """

    processed_data_file: Path
    """File in which to save the processed data"""
