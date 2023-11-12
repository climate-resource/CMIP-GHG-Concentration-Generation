"""
Config for the analysis branch
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class AnalysisConfig:
    """
    Configuration class for the analysis branch
    """

    branch_config_id: str
    """
    ID for this configuration of the branch

    Must be unique among all configurations for this branch
    """

    mean_dir: Path
    """Directory in which to save text files of the means in the x-direction"""
