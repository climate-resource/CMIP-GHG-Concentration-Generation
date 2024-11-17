"""
Config for the quick crunch
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class QuickCrunchConfig:
    """
    Configuration class for the quick crunch
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    processed_data_file_global_means: Path
    """Path in which to save the processed global-means"""
