"""
Config for the plot step
"""

from __future__ import annotations

from attrs import frozen


@frozen
class PlotConfig:
    """
    Configuration class for the plot step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """
