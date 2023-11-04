"""
Config for the preparation branch
"""
from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class PreparationConfig:
    """
    Configuration class for the preparation branch
    """

    branch_config_id: str
    """
    ID for this configuration of the branch

    Must be unique among all configurations for this branch
    """

    seed: int
    """Seed to use for random draws"""

    seed_file: Path
    """Path in which to save the seed"""
