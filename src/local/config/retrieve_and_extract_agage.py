"""
Config for retrieving and extracting AGAGE data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource


@frozen
class RetrieveExtractAGAGEDataConfig:
    """
    Configuration class for retrieving and extracting AGAGE data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas for which we are processing data"""

    instrument: str
    """Instrument for which to retrieve data"""

    time_frequency: str
    """Time frequency to retrieve"""

    download_urls: list[URLSource]
    """URLs from which to download the data"""

    raw_dir: Path
    """Directory in which to save the raw data"""

    readme: URLSource
    """Source from which to get the README"""

    download_complete_file: Path
    """
    Path in which to write the time at which the download was completed

    This is mainly used to help with setting the dependencies between notebooks correctly.
    """

    processed_monthly_data_with_loc_file: Path
    """
    Path in which to write the processed monthly data including location information
    """

    source_info_short_names_file: Path
    """Path in which to save the source info short names for this retrieval"""

    generate_hashes: bool
    """
    Should we generate the hashes for the files?

    If yes, these will be printed in the executed notebook.
    For production runs, this should be set to False.
    """
