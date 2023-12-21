"""
Config for the process step
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class ProcessConfig:
    """
    Configuration class for the process step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    law_dome: LawDomeConfig
    """Configuration for processing Law Dome data"""

    gggrn: GGGRNConfig
    """Configuration for the global greenhouse gas reference network"""


@frozen
class LawDomeConfig:
    """
    Configuration class for Law Dome data
    """

    processed_file: Path
    """File in which to save the processed data"""


@frozen
class GGGRNConfig:
    """
    Configuration class for the Global Greenhouse Gas Reference Network (GGGRN)
    """

    processed_file_global_mean: Path
    """File in which to save the processed global-mean data"""
