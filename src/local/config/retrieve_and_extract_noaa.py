"""
Config for retrieving and extracting NOAA data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource


@frozen
class RetrieveExtractNOAADataConfig:
    """
    Configuration class for retrieving and extracting NOAA data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas for which we are processing data"""

    source: str
    """Source (i.e. network e.g. surface flask network) of the data we are processing"""

    download_urls: list[URLSource]
    """URLs from which to download the data"""

    raw_dir: Path
    """Directory in which to save the raw data"""

    download_complete_file: Path
    """
    Path in which to write the time at which the download was completed

    This is mainly used to help with setting the dependencies between notebooks
    correctly.
    """

    interim_files: dict[str, Path]
    """
    Where to save interim files

    This is a very flexible container so you can basically put whatever in here that
    doesn't fit with the wider pattern.
    """
