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

    law_dome: LawDomeConfig
    """Configuration for retrieving Law Dome data"""

    gggrn: GGGRNConfig
    """Configuration for the global greenhouse gas reference network"""


@frozen
class LawDomeConfig:
    """
    Configuration class for Law Dome data
    """

    doi: str
    """DOI of the dataset"""

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
