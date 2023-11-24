"""
Config for the preparation step
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class PreparationConfig:
    """
    Configuration class for the preparation step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    seed: int
    """Seed to use for random draws"""

    seed_file: Path
    """Path in which to save the seed"""
