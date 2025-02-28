"""
Config for retrieving, extracting and processing Western et al. (2024) data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource

from local.dependencies import SourceInfo


@frozen
class RetrieveExtractWesternEtal2024Data:
    """
    Config for retrieving, extracting and processing Western et al. (2024) data

    Original paper: https://doi.org/10.1038/s41558-024-02038-7
    Zenodo record: https://zenodo.org/records/10782689
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
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

    source_info: SourceInfo
    """Source information"""
