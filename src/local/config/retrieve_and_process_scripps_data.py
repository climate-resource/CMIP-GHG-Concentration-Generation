"""
Config for retrieving, extracting and processing Scripps data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource

from local.dependencies import SourceInfo


@frozen
class RetrieveProcessScrippsConfig:
    """
    Configuration class for retrieving and processing Scripps data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    merged_ice_core_data: URLSource
    """URLs from which to download the merged ice core data"""

    merged_ice_core_data_processed_data_file: Path
    """Path in which to save the processed merged ice core data"""

    merged_ice_core_data_source_info: SourceInfo
    """Source information"""

    station_data: list[ScrippsSource]
    """Information about the station data"""

    raw_dir: Path
    """
    Directory in which the raw data is saved
    """

    processed_data_with_loc_file: Path
    """File in which to save the processed data, including location information"""


@frozen
class ScrippsSource:
    """Information about a Scripps station"""

    url_source: URLSource
    """URL from which to download the data"""

    station_code: str
    """Station code"""

    lat: str
    """Latitude"""

    lon: str
    """Longitude"""
