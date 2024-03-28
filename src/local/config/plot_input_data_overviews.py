"""
Config for the input data overview plotting step
"""

from __future__ import annotations

from attrs import frozen


@frozen
class PlotInputDataOverviewsConfig:
    """
    Configuration class for the input data overview plotting step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """
