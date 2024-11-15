"""
Lazy linear mean-preserving interpolator
"""

from __future__ import annotations

import pint
from attrs import define


@define
class LazyLinearInterpolator:
    """
    Lazy, linear mean-preserving interpolator

    This class' algorithm is very basic.
    It simply performs linear interpolation,
    then adjusts the values in each interval to match the input values' mean,
    without regard for the other values in the array.
    As a result, it can produce output which is highly discontinuous
    so may be inapproapriate for some use cases.

    Really, this algorithm is only implemented for testing purposes.
    We do not recommend using it in production.
    """

    def __call__(
        self,
        x_bounds_in: pint.UnitRegistry.Quantity,
        y_in: pint.UnitRegistry.Quantity,
        x_bounds_out: pint.UnitRegistry.Quantity,
    ) -> pint.UnitRegistry.Quantity:
        pass
