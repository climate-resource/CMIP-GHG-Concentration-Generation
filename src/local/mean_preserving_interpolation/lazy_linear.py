"""
Lazy linear mean-preserving interpolator
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pint
from attrs import define

from local.mean_preserving_interpolation.grouping import (
    get_group_averages,
    get_group_indexes,
)


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
        """
        Perform mean-preserving interpolation

        Parameters
        ----------
        x_bounds_in
            Bounds of the x-range to which each value in `y_in` applies.

        y_in
            y-values for each interval in `x_bounds_in`.

        x_bounds_out
            Bounds of the x-values onto which to interpolate `y_in`.

        Returns
        -------
        :
            Interpolated, mean-preserving values
        """
        x_mid_points_in = (x_bounds_in[1:] + x_bounds_in[:-1]) / 2.0
        x_mid_points_out = (x_bounds_out[1:] + x_bounds_out[:-1]) / 2.0

        raw_interp = cast(
            pint.UnitRegistry.Quantity,
            np.interp(x_mid_points_out, x_mid_points_in, y_in),
        )

        raw_means = get_group_averages(
            integrand_x_bounds=x_bounds_out,
            integrand_y=raw_interp,
            group_bounds=x_bounds_in,
        )

        diff_from_input = y_in - raw_means
        group_idx = get_group_indexes(x_bounds=x_bounds_out, group_bounds=x_bounds_in)
        adjustments = diff_from_input[(group_idx,)]

        res = cast(pint.UnitRegistry.Quantity, raw_interp + adjustments)

        return res
