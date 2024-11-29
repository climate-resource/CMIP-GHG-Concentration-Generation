"""
Config for retrieving, extracting and processing Velders et al. (2022) data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource


@frozen
class RetrieveExtractVeldersEtal2022Data:
    """
    Config for retrieving, extracting and processing Velders et al. (2022) data

    Original paper: https://doi.org/10.5194/acp-22-6087-2022
    Zenodo record: https://zenodo.org/records/6520707
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    raw_data_file_tmp: Path
    """
    File in which to the raw data is found

    This is a temporary thing while we wait for Guus to upload the Zenodo record.
    """

    zenodo_record: URLSource
    """Zenodo record from which to download the raw data"""

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
