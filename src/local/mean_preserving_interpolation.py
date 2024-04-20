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

    alpha, knots, degree = mean_preserving_interpolation(
        X=X,
        Y=Y,
        x=x,
    )

    def interpolator(x):
        return Quantity(
            scipy.interpolate.BSpline(t=knots, c=alpha, k=degree)(x),
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


def mean_preserving_interpolation(
    X: np.ndarray,
    Y: np.ndarray,
    x: np.ndarray,
    degree: int = 3,
    alpha: float = 1.01,
) -> tuple[np.ndarray, np.ndarray, int]:
    resolution_increase = int(x.size / X.size)

    degrees_freedom = int(np.ceil(alpha * Y.size))

    knots_prev = np.repeat(x[0], degree)
    knots_post = np.repeat(x[-1], degree)
    knots_internal = np.quantile(x, np.linspace(0, 1, degrees_freedom - degree))
    knots = np.hstack([knots_prev, knots_internal, knots_post])

    alpha_len = knots.size - degree - 1

    B = np.vstack(
        [
            # Nicolai had the line below, I can't understand why and including it breaks things
            # np.ones(Y.size),
            scipy.interpolate.BSpline.design_matrix(x, t=knots, k=degree).toarray(),
        ]
    )
    if alpha_len != B.shape[1]:
        raise AssertionError

    BM = np.zeros((Y.size, B.shape[1]))
    for i in range(Y.size):
        BM[i, :] = np.mean(
            B[i * resolution_increase : (i + 1) * resolution_increase], axis=0
        )

    BD = np.diff(B, axis=0)

    c = np.hstack([np.ones(BD.shape[0]), np.zeros(2 * alpha_len)])

    A_eq = np.hstack([np.zeros((Y.size, BD.shape[0])), BM, -BM])
    b_eq = Y
    b_ub = np.zeros(2 * BD.shape[0])
    A_ub = np.vstack(
        [
            np.hstack([-np.eye(BD.shape[0]), BD, -BD]),
            np.hstack([-np.eye(BD.shape[0]), -BD, BD]),
        ]
    )

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
        raise AssertionError

    alpha = res.x[-2 * alpha_len : -alpha_len] - res.x[-alpha_len:]

    return alpha, knots, degree
