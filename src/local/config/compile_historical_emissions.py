"""
Config for the compilation of historical emissions data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class CompileHistoricalEmissionsConfig:
    """
    Configuration class for the compilation of historical emissions data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    complete_historical_emissions_file: Path
    """Path in which to save the complete historical emissions"""
