"""
Mean-preserving interpolation algorithms
"""

from __future__ import annotations

import cftime
import numpy as np
import pint
import pint.testing
import scipy.interpolate
import scipy.optimize
import xarray as xr

from local.xarray_space import calculate_global_mean_from_lon_mean
from local.xarray_time import convert_time_to_year_month

Quantity = pint.get_application_registry().Quantity


def interpolate_annual_mean_to_monthly(annual_mean: xr.DataArray) -> xr.DataArray:
    X = annual_mean["year"].data
    Y = annual_mean.data.m
    # These are monthly timesteps, centred in the middle of each month
    N_MONTHS_PER_YEAR = 12
    x = (
        np.arange(np.floor(np.min(X)), np.ceil(np.max(X) + 1), 1 / N_MONTHS_PER_YEAR)
        + 1 / N_MONTHS_PER_YEAR / 2
    )

    coefficients, intercept, knots, degree = mean_preserving_interpolation(
        X=X,
        Y=Y,
        x=x,
    )

    def interpolator(x):
        return Quantity(
            scipy.interpolate.BSpline(t=knots, c=coefficients, k=degree)(x) + intercept,
            annual_mean.data.units,
        )

    y = interpolator(x)

    time = [
        cftime.datetime(
            np.floor(time_val),
            np.round(N_MONTHS_PER_YEAR * (time_val % 1 + 1 / N_MONTHS_PER_YEAR / 2)),
            1,
        )
        for time_val in x
    ]

    out = xr.DataArray(
        data=y,
        dims=["time"],
        coords=dict(time=time),
    )

    pint.testing.assert_allclose(
        out.groupby("time.year").mean().data,
        annual_mean.data.squeeze(),
    )

    return convert_time_to_year_month(out)


def interpolate_lat_15_degree_to_half_degree(
    lat_15_degree: xr.DataArray,
) -> xr.DataArray:
    ASSUMED_INPUT_LAT_SPACING = 15
    TARGET_LAT_SPACING = 0.5

    ASSUMED_LAT_BINS = np.arange(-90, 91, ASSUMED_INPUT_LAT_SPACING)
    ASSUMED_INPUT_LAT_CENTRES = np.mean(
        np.vstack([ASSUMED_LAT_BINS[1:], ASSUMED_LAT_BINS[:-1]]), axis=0
    )
    np.testing.assert_allclose(lat_15_degree["lat"].data, ASSUMED_INPUT_LAT_CENTRES)

    X = lat_15_degree["lat"].data
    Y = lat_15_degree.data.m

    x = np.arange(
        np.min(X) - ASSUMED_INPUT_LAT_SPACING / 2 + TARGET_LAT_SPACING / 2,
        np.max(X) + ASSUMED_INPUT_LAT_SPACING / 2 + TARGET_LAT_SPACING / 2,
        TARGET_LAT_SPACING,
    )

    # TODO: split out get_lat_weights function
    # Also re-think. This function assumes that our quantities apply to the whole
    # cell, whereas ours probably only apply to the centre of the cell (points)
    # hence cos weighting probably better.
    x_lb = Quantity(x - TARGET_LAT_SPACING / 2, "degrees_north")
    x_ub = Quantity(x_lb.m + TARGET_LAT_SPACING, "degrees_north")
    weights = (
        (np.sin(x_ub.to("radian")) - np.sin(x_lb.to("radian")))
        .to("dimensionless")
        .m.squeeze()
    )

    coefficients, intercept, knots, degree = mean_preserving_interpolation(
        X=X,
        Y=Y,
        x=x,
        weights=weights,
    )

    def interpolator(x):
        return Quantity(
            scipy.interpolate.BSpline(t=knots, c=coefficients, k=degree)(x) + intercept,
            lat_15_degree.data.units,
        )

    y = interpolator(x)

    out = xr.DataArray(
        name="fine_grid",
        data=y,
        dims=["lat"],
        coords=dict(lat=x),
    )

    pint.testing.assert_allclose(
        out.groupby_bins("lat", ASSUMED_LAT_BINS)
        .apply(calculate_global_mean_from_lon_mean)
        .data.squeeze(),
        lat_15_degree.data.squeeze(),
    )

    return out


def mean_preserving_interpolation(
    X: np.ndarray,
    Y: np.ndarray,
    x: np.ndarray,
    degree: int = 3,
    degrees_freedom_scalar: float = 1.01,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if weights is None:
        weights = np.ones_like(x)

    resolution_increase = int(x.size / X.size)

    degrees_freedom = int(np.ceil(degrees_freedom_scalar * Y.size))

    knots_prev = np.repeat(x[0], degree)
    knots_post = np.repeat(x[-1], degree)
    knots_internal = np.quantile(x, np.linspace(0, 1, degrees_freedom - degree))
    knots = np.hstack([knots_prev, knots_internal, knots_post])

    alpha_len = knots.size - degree

    B = np.column_stack(
        [
            np.ones(x.size),
            scipy.interpolate.BSpline.design_matrix(x, t=knots, k=degree).toarray(),
        ]
    )
    if alpha_len != B.shape[1]:
        raise AssertionError

    BM = np.zeros((X.size, B.shape[1]))
    for i in range(X.size):
        start_idx = i * resolution_increase
        stop_idx = (i + 1) * resolution_increase
        BM[i, :] = np.average(
            B[start_idx:stop_idx, :], axis=0, weights=weights[start_idx:stop_idx]
        )

    BD = np.diff(B, axis=0)

    c = np.hstack([np.ones(BD.shape[0]), np.zeros(2 * alpha_len)])

    A_eq = np.column_stack([np.zeros((Y.size, BD.shape[0])), BM, -BM])
    b_eq = Y
    A_ub = np.column_stack(
        [
            np.hstack([-np.eye(BD.shape[0]), BD, -BD]),
            np.hstack([-np.eye(BD.shape[0]), -BD, BD]),
        ]
    )
    b_ub = np.zeros(2 * BD.shape[0])

    res = scipy.optimize.linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        method="highs",
        bounds=(0, None),
        # options=dict(maxiter=int(maxiter)),
    )
    if not res.success:
        raise AssertionError(res.message)

    alpha = res.x[-2 * alpha_len : -alpha_len] - res.x[-alpha_len:]
    intercept = alpha[0]
    coefficients = alpha[1:]

    print(f"{intercept=}")

    return coefficients, intercept, knots, degree
