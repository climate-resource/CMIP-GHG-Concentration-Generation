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


@frozen
class LawDomeConfig:
    """
    Configuration class for Law Dome data
    """

    doi: str
    """DOI of the dataset"""

    files_md5_sum: dict[Path, str]
    """MD5 hashes for the files in the dataset"""
