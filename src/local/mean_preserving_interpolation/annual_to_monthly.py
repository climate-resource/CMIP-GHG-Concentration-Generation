"""
Mean-preserving interpolation of yearly values down to monthly values
"""

from __future__ import annotations

import cftime
import numpy as np
import pint
import xarray as xr

from local.mean_preserving_interpolation.core import (
    MeanPreservingInterpolationAlgorithmLike,
    mean_preserving_interpolation,
)
from local.mean_preserving_interpolation.lai_kaplan import (
    LaiKaplanInterpolator,
    get_wall_control_points_y_linear_with_flat_override_on_left,
)
from local.xarray_time import convert_time_to_year_month

Q = pint.get_application_registry().Quantity

N_MONTHS_PER_YEAR: int = 12
"""Number of months in a year"""


def interpolate_annual_mean_to_monthly(
    annual_mean: xr.DataArray,
    algorithm: str | MeanPreservingInterpolationAlgorithmLike = LaiKaplanInterpolator(
        get_wall_control_points_y_from_interval_ys=get_wall_control_points_y_linear_with_flat_override_on_left,
    ),
    month_rounding: int = 4,
    verify_output_is_mean_preserving: bool = True,
) -> xr.DataArray:
    """
    Interpolate annual-mean to monthly values, preserving the annual-mean

    Parameters
    ----------
    annual_mean
        Annual-mean value to interpolate

    algorithm
        Algorithm to use for the interpolation

    month_rounding
        Rounding to apply to the monthly float values.

        Unlikely that you'll need to change this.

    verify_output_is_mean_preserving
        Whether to verify that the output is mean-preserving before returning.

    Returns
    -------
    :
        Monthly interpolated values
    """
    y_in = annual_mean.data
    Quantity = y_in.u._REGISTRY.Quantity
    yrs = annual_mean.year.to_numpy()
    x_bounds_in = Quantity(np.hstack([yrs, yrs[-1] + 1.0]), "yr")
    x_bounds_out = Quantity(
        np.round(
            np.arange(x_bounds_in[0].m, x_bounds_in[-1].m + 1 / N_MONTHS_PER_YEAR / 2, 1 / N_MONTHS_PER_YEAR),
            month_rounding,
        ),
        "yr",
    )

    monthly_vals = mean_preserving_interpolation(
        x_bounds_in=x_bounds_in,
        y_in=y_in,
        x_bounds_out=x_bounds_out,
        algorithm=algorithm,
        verify_output_is_mean_preserving=verify_output_is_mean_preserving,
    )

    month_out = (x_bounds_out[1:] + x_bounds_out[:-1]) / 2.0

    time_out = [
        cftime.datetime(
            np.floor(time_val),
            np.round(N_MONTHS_PER_YEAR * (time_val % 1 + 1 / N_MONTHS_PER_YEAR / 2)),
            1,
        )
        for time_val in month_out.m
    ]

    out_time = xr.DataArray(
        data=monthly_vals,
        dims=["time"],
        coords=dict(time=time_out),
    )
    out = convert_time_to_year_month(out_time)

    return out
