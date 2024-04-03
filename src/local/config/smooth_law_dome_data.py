"""
Config for smoothing the Law Dome data
"""

from __future__ import annotations

import pint
from attrs import frozen

from local.noise_addition import NoiseAdderPercentageXNoise


@frozen
class SmoothLawDomeDataConfig:
    """
    Configuration class for smoothing of the Law Dome data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    gas: str
    """Gas being smoothed"""

    noise_adder: NoiseAdderPercentageXNoise
    """Noise adder to use during the smoothing"""

    point_selector_settings: PointSelectorSettings
    """Settings to use when creating our point selector"""


@frozen
class PointSelectorSettings:
    """
    Settings to use when creating our point selector
    """

    window_width: pint.UnitRegistry.Quantity
    """Window width to use"""

    minimum_data_points_either_side: int
    """Minimum number of data points to use on either side of the target point"""

    maximum_data_points_either_side: int
    """Maximum number of data points to use on either side of the target point"""
