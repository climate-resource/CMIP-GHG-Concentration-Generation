"""
Config for the retrieve branch
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class RetrieveConfig:
    """
    Configuration class for the retrieve branch
    """

    branch_config_id: str
    """
    ID for this configuration of the branch

    Must be unique among all configurations for this branch
    """

    law_dome: LawDomeConfig
    """Configuration for retrieving law dome data"""


@frozen
class LawDomeConfig:
    """
    Configuration class for law dome data
    """

    doi: str
    """DOI of the dataset"""

    files_md5_sum: dict[Path, str]
    """MD5 hashes for the files in the dataset"""
