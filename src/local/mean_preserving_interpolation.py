"""
Mean-preserving interpolation algorithms
"""

from __future__ import annotations

from typing import TypeVar

import cftime
import numpy as np
import numpy.typing as npt
import pint
import pint.testing
import scipy.interpolate
import scipy.optimize
import xarray as xr

from local.xarray_space import calculate_global_mean_from_lon_mean
from local.xarray_time import convert_time_to_year_month

T = TypeVar("T")


def interpolate_annual_mean_to_monthly(
    annual_mean: xr.DataArray,
    degrees_freedom_scalar: float = 1.1,
) -> xr.DataArray:
    """
    Interpolate annual-mean values to monthly values.

    The interpolation preserves the annual-mean.

    Parameters
    ----------
    annual_mean
        Annual-mean values to interpolate.

    degrees_freedom_scalar
        Degrees of freedom to use when calculating the interpolating spline.

    Returns
    -------
        Values, interpolated onto a monthly time axis.
    """
    Quantity = pint.get_application_registry().Quantity
    X = annual_mean["year"].data.squeeze()
    Y = annual_mean.data.m.squeeze()

    if len(X.shape) != 1:
        msg = f"Squeezed X must be 1D, received {X.shape=}"
        raise AssertionError(msg)

    if len(Y.shape) != 1:
        msg = f"Squeezed Y must be 1D, received {Y.shape=}"
        raise AssertionError(msg)

    # These are monthly timesteps, centred in the middle of each month
    N_MONTHS_PER_YEAR = 12
    # TODO: speak with Nicolai about how to boundary counditions better.
    # The below is a hack to try and get slightly more sensible behaviour at the boundaries.
    # It just does basic linear extrapolation at the boundaries.
    X = np.hstack([2 * X[0] - X[1], X, 2 * X[-1] - X[-2]])
    Y = np.hstack([2 * Y[0] - Y[1], Y, 2 * Y[-1] - Y[-2]])
    x = (
        np.arange(np.floor(np.min(X)), np.ceil(np.max(X)) + 1, 1 / N_MONTHS_PER_YEAR)
        + 1 / N_MONTHS_PER_YEAR / 2
    )

    coefficients, intercept, knots, degree = mean_preserving_interpolation(
        X=X,
        Y=Y,
        x=x,
        degrees_freedom_scalar=degrees_freedom_scalar,
    )

    # Undo hack above
    x = x[N_MONTHS_PER_YEAR:-N_MONTHS_PER_YEAR]

    def interpolator(xh):
        return Quantity(
            scipy.interpolate.BSpline(t=knots, c=coefficients, k=degree)(xh)
            + intercept,
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
        annual_mean.squeeze().data,
    )

    return convert_time_to_year_month(out)


def interpolate_lat_15_degree_to_half_degree(
    lat_15_degree: xr.DataArray,
    degrees_freedom_scalar: float = 1.75,
) -> xr.DataArray:
    """
    Interpolate data on a 15 degree latitudinal grid to a 0.5 degree latitudinal grid.

    Parameters
    ----------
    lat_15_degree
        Data on a 15 degree latitudinal grid

    degrees_freedom_scalar
        Degrees of freedom to use in the interpolation

    Returns
    -------
        Data interpolated onto a 0.5 degree latitudinal grid.
        The interpolation reflects the area-weighted mean of ``lat_15_degree``.
    """
    Quantity = pint.get_application_registry().Quantity
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

    assert False, "fix this"
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
        degrees_freedom_scalar=degrees_freedom_scalar,
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


def mean_preserving_interpolation(  # noqa: PLR0913
    X: np.ndarray,
    Y: np.ndarray,
    x: np.ndarray,
    degrees_freedom_scalar: float,
    degree: int = 3,
    weights: np.ndarray | None = None,
) -> tuple[npt.NDArray[np.float64], np.float64, npt.NDArray[np.float64], int]:
    """
    Perform a mean-preserving interpolation

    Parameters
    ----------
    X
        x-values of the input

    Y
        y-values of the input

    x
        x-values of the target x-grid

    degrees_freedom_scalar
        Degrees of freedom to use when creating our interpolating spline

    degree
        Degree of the interpolating spline (3, the default, is a cubic spline)

    weights
        Weights to apply to each point in x when calculating the mean.


    Returns
    -------
        The coeffecients, intercept, knots and degree of the interpolating B-spline.
        This can be turned into an interpolating function using
        ``scipy.interpolate.BSpline(t=knots, c=coefficients, k=degree)(x) + intercept``.
    """
    if weights is None:
        weights = np.ones_like(x)

    resolution_increase = int(x.size / X.size)

    degrees_freedom = int(np.ceil(degrees_freedom_scalar * Y.size))

    knots_prev = np.repeat(x[0], degree)
    knots_post = np.repeat(x[-1], degree)
    knots_internal = np.quantile(x, np.linspace(0, 1, degrees_freedom - degree + 1))
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

    A_eq = np.column_stack([np.zeros((X.size, BD.shape[0])), BM, -BM])
    b_eq = Y
    A_ub = np.row_stack(
        [
            np.column_stack([-np.eye(BD.shape[0]), BD, -BD]),
            np.column_stack([-np.eye(BD.shape[0]), -BD, BD]),
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

    return coefficients, intercept, knots, degree


def interpolate_time_slice_parallel_helper(
    inp: tuple[T, xr.DataArray]
) -> tuple[T, xr.DataArray]:
    """
    Interpolate time slice in parallel.

    This is a helper function that makes the parallelisation possible.
    It applies {py:func}`interpolate_lat_15_degree_to_half_degree`
    to the given time slice.

    Parameters
    ----------
    inp
        Input values. The first element should be the time to which this slice applies.
        The second is the time slice to interpolate.

    Returns
    -------
        The time slice to which this slice applies
        and the interpolated time slice.
    """
    time, da = inp
    import cf_xarray.units
    import pint_xarray

    cf_xarray.units.units.define("ppm = 1 / 1000000")
    cf_xarray.units.units.define("ppb = ppm / 1000")
    cf_xarray.units.units.define("ppt = ppb / 1000")

    pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
        cf_xarray.units.units
    )
    pint.set_application_registry(pint_xarray.accessors.default_registry)

    return time, interpolate_lat_15_degree_to_half_degree(da.pint.quantify())
