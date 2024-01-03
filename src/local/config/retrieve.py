"""
Config for the retrieve step
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class RetrieveConfig:
    """
    Configuration class for the retrieve step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    noaa_network: NOAANetworkConfig
    """Configuration for retrieving data from the NOAA network"""

    law_dome: LawDomeConfig
    """Configuration for retrieving Law Dome data"""

    gggrn: GGGRNConfig
    """Configuration for the global greenhouse gas reference network"""

    natural_earth: NaturalEarthConfig
    """Configuration for retrieving data from natural earth"""


@frozen
class NOAANetworkConfig:
    """
    Configuration class for the NOAA network data
    """

    raw_dir: Path
    """Directory in which to save the raw data"""

    download_urls: list[URLSource]
    """URLs from which to download data"""


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
class LawDomeConfig:
    """
    Configuration class for Law Dome data
    """

    doi: str
    """DOI of the dataset"""

    raw_dir: Path
    """Directory in which to save the raw data"""

    files_md5_sum: dict[Path, str]
    """MD5 hashes for the files in the dataset"""


@frozen
class GGGRNConfig:
    """
    Configuration class for the Global Greenhouse Gas Reference Network (GGGRN)
    """

    raw_dir: Path
    """Directory in which to save the raw data"""

    urls_global_mean: list[URLSource]
    """URLs from which to download the raw global-mean data"""


@frozen
class URLSource:
    """
    Source information for downloading a source from a URL
    """

    url: str
    """URL to download from"""

    known_hash: str
    """Known hash for the downloaded file"""
