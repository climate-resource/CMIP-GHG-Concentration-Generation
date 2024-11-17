"""
Mean preserving interpolation

This is a surprisingly tricky thing to do.
Hence, this module is surprisingly large.
"""

from __future__ import annotations

from local.mean_preserving_interpolation.annual_to_monthly import interpolate_annual_mean_to_monthly
from local.mean_preserving_interpolation.core import (
    MeanPreservingInterpolationAlgorithmLike,
    mean_preserving_interpolation,
)

__all__ = [
    "MeanPreservingInterpolationAlgorithmLike",
    "interpolate_annual_mean_to_monthly",
    "mean_preserving_interpolation",
]
