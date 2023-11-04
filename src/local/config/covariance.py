"""
Config for the covariance branch
"""
from __future__ import annotations

import numpy as np
import numpy.typing as nptype
from attrs import frozen


@frozen
class CovarianceConfig:
    """
    Configuration class for the covariance branch
    """

    branch_config_id: str
    """
    ID for this configuration of the branch

    Must be unique among all configurations for this branch
    """

    covariance: nptype.NDArray[np.float64]
    """Covariance to use when making draws"""
