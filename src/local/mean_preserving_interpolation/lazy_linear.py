"""
Lazy linear mean-preserving interpolator
"""

from __future__ import annotations

import numpy as np
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

        raw_interp = np.interp(x_mid_points_out, x_mid_points_in, y_in)

        raw_means = integrate_groups(
            integrand_x_bounds=x_bounds_out,
            integrand_y=raw_interp,
            group_bounds=x_bounds_in,
        )
        # breakpoint()
        x_out_size = x_bounds_out[1:] - x_bounds_out[:-1]
        x_in_size = x_bounds_in[1:] = x_bounds_in[:-1]

        res_increase = (x_bounds_out.size - 1) / (x_bounds_in.size - 1)
        res_increase = int(res_increase)

        x_out_integrals = x_out_size * raw_interp
        tmp = np.cumsum(x_out_integrals)[::res_increase]

        np.repeat(x_in_size.m, res_increase) * x_in_size.u
