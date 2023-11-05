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

    branch_config_id: str
    """
    ID for this configuration of the branch

    Must be unique among all configurations for this branch
    """
