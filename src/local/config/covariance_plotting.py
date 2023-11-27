"""
Config for the covariance plotting
"""
from __future__ import annotations

from attrs import frozen


@frozen
class CovariancePlottingConfig:
    """
    Configuration for the covariance plotting
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """
