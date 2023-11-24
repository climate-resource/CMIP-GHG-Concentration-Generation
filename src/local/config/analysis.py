"""
Config for the analysis step
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class AnalysisConfig:
    """
    Configuration class for the analysis step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    mean_dir: Path
    """Directory in which to save text files of the means in the x-direction"""
