"""
Config for the retrieve step
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource


@frozen
class RetrieveMiscDataConfig:
    """
    Configuration class for the retrieve step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    natural_earth: NaturalEarthConfig
    """Configuration for retrieving data from natural earth"""

    primap: PRIMAPConfig
    """Configuration for retrieving data from PRIMAP"""

    hadcrut5: HadCRUT5Config
    """Configuration for retrieving data from HadCRUT5"""


@frozen
class NaturalEarthConfig:
    """
    Configuration class for retrieving natural earth data
    """

    raw_dir: Path
    """Directory in which to save the raw data"""

    download_urls: list[URLSource]
    """URLs from which to download data"""

    countries_shape_file_name: str
    """Name of the file containing shape files for country borders"""


@frozen
class PRIMAPConfig:
    """
    Configuration class for retrieving PRIMAP data
    """

    raw_dir: Path
    """Directory in which to save the raw data"""

    download_url: URLSource
    """URL from which to download data"""


@frozen
class HadCRUT5Config:
    """
    Configuration class for retrieving HadCRUT5 data
    """

    raw_dir: Path
    """Directory in which to save the raw data"""

    download_url: URLSource
    """URL from which to download data"""
