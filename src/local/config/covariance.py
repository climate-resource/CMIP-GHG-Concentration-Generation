"""
Config for the covariance step
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as nptype
from attrs import frozen


@frozen
class CovarianceConfig:
    """
    Configuration class for the covariance step
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    covariance: nptype.NDArray[np.float64]
    """Covariance to use when making draws"""

    draw_file: Path
    """Path in which to save the drawn data"""
