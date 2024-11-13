"""
Mean-preserving interpolation algorithms
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Generic, TypeVar

import attrs.validators
import cftime
import numpy as np
import numpy.typing as npt
import pint
import pint.testing
import scipy.interpolate  # type: ignore
import scipy.optimize  # type: ignore
import xarray as xr
from attrs import define, field
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from local.xarray_time import convert_time_to_year_month

T = TypeVar("T")


def interpolate_annual_mean_to_monthly(
    annual_mean: xr.DataArray,
    degrees_freedom_scalar: float = 1.1,
    rtol: float = 1e-8,
    atol: float = 5e-6,
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

    rtol
        Relative tolerance to apply
        while checking the interpolation preserved the annual-mean.

    atol
        Absolute tolerance to apply
        while checking the interpolation preserved the annual-mean.
        This should line up with the tolerance that is used by {py:func}`mean_preserving_interpolation`.

    Returns
    -------
        Values, interpolated onto a monthly time axis.
    """
    Quantity = pint.get_application_registry().Quantity  # type: ignore
    # put in middle of year
    X = annual_mean["year"].data.squeeze() + 0.5
    Y = annual_mean.data.m.squeeze()

    if len(X.shape) != 1:
        msg = f"Squeezed X must be 1D, received {X.shape=}"
        raise AssertionError(msg)

    if len(Y.shape) != 1:
        msg = f"Squeezed Y must be 1D, received {Y.shape=}"
        raise AssertionError(msg)

    # These are monthly timesteps, centred in the middle of each month
    N_MONTHS_PER_YEAR = 12
    # # TODO: speak with Nicolai about how to boundary counditions better.
    # # The below is a hack to try and get slightly more sensible behaviour at the boundaries.
    # # It just does basic linear extrapolation at the boundaries.
    # X = np.hstack([2 * X[0] - X[1], X, 2 * X[-1] - X[-2]])
    # Y = np.hstack([2 * Y[0] - Y[1], Y, 2 * Y[-1] - Y[-2]])

    x = (
        np.arange(np.floor(np.min(X)), np.ceil(np.max(X)), 1 / N_MONTHS_PER_YEAR)
        + 1 / N_MONTHS_PER_YEAR / 2
    )

    interpolator_raw = mean_preserving_interpolation(
        X=X,
        Y=Y,
        x=x,
        degrees_freedom_scalar=degrees_freedom_scalar,
    )

    def interpolator(
        xh: float | int | npt.NDArray[np.float64],
    ) -> pint.UnitRegistry.Quantity:
        return Quantity(  # type: ignore
            interpolator_raw(xh),  # type: ignore
            annual_mean.data.units,
        )

    # coefficients, intercept, knots, degree = mean_preserving_interpolation(
    #     X=X,
    #     Y=Y,
    #     x=x,
    #     degrees_freedom_scalar=degrees_freedom_scalar,
    # )
    #
    # # Undo hack above
    # x = x[N_MONTHS_PER_YEAR:-N_MONTHS_PER_YEAR]
    #
    # def interpolator(
    #     xh: float | int | npt.NDArray[np.float64],
    # ) -> pint.UnitRegistry.Quantity:
    #     return Quantity(  # type: ignore
    #         scipy.interpolate.BSpline(t=knots, c=coefficients, k=degree)(xh) + intercept,
    #         annual_mean.data.units,
    #     )

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

    # # TODO: turn it back into mean-preserving
    # pint.testing.assert_allclose(
    #     out.groupby("time.year").mean().data,
    #     annual_mean.squeeze().data,
    #     rtol=rtol,
    #     atol=atol,
    # )

    return convert_time_to_year_month(out)


def interpolate_lat_15_degree_to_half_degree(
    lat_15_degree: xr.DataArray,
    degrees_freedom_scalar: float = 1.75,
    atol: float = 5e-6,
) -> xr.DataArray:
    """
    Interpolate data on a 15 degree latitudinal grid to a 0.5 degree latitudinal grid.

    Parameters
    ----------
    lat_15_degree
        Data on a 15 degree latitudinal grid

    degrees_freedom_scalar
        Degrees of freedom to use in the interpolation

    atol
        Absolute tolerance for checking consistency with input data,
        and whether the input data is zero.
        This should line up with the tolerance that is used by {py:func}`mean_preserving_interpolation`.

    Returns
    -------
        Data interpolated onto a 0.5 degree latitudinal grid.
        The interpolation reflects the area-weighted mean of ``lat_15_degree``.
    """
    Quantity = pint.get_application_registry().Quantity  # type: ignore
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

    if np.isclose(Y, 0.0, atol=atol).all():
        # Short-cut
        y = Quantity(np.zeros_like(x), lat_15_degree.data.units)

    else:
        # Assume that each value just applies to a point
        # (no area extent/interpolation of the data,
        # i.e. we're calculating a weighted sum, not an integral).
        # Hence use cos here.
        # weights = np.cos(np.deg2rad(x))

        interpolator_raw = mean_preserving_interpolation(
            X=X,
            Y=Y,
            x=x,
            degrees_freedom_scalar=degrees_freedom_scalar,
        )

        def interpolator(
            xh: float | int | npt.NDArray[np.float64],
        ) -> pint.UnitRegistry.Quantity:
            return Quantity(  # type: ignore
                interpolator_raw(xh),  # type: ignore
                lat_15_degree.data.units,
            )

        # coefficients, intercept, knots, degree = mean_preserving_interpolation(
        #     X=X,
        #     Y=Y,
        #     x=x,
        #     weights=weights,
        #     degrees_freedom_scalar=degrees_freedom_scalar,
        # )
        #
        # def interpolator(
        #     x: float | int | npt.NDArray[np.float64],
        # ) -> pint.UnitRegistry.Quantity:
        #     return Quantity(  # type: ignore
        #         scipy.interpolate.BSpline(t=knots, c=coefficients, k=degree)(x) + intercept,
        #         lat_15_degree.data.units,
        #     )

        y = interpolator(x)

    out = xr.DataArray(
        name="fine_grid",
        data=y,
        dims=["lat"],
        coords=dict(lat=x),
    )

    # # TODO: turn it back into mean-preserving
    # pint.testing.assert_allclose(
    #     out.groupby_bins("lat", ASSUMED_LAT_BINS)  # type: ignore
    #     .apply(calculate_global_mean_from_lon_mean)
    #     .data.squeeze(),
    #     lat_15_degree.data.squeeze(),
    #     atol=atol,
    # )

    return out


# Paper reference: https://journals.ametsoc.org/view/journals/atot/39/4/JTECH-D-21-0154.1.xml#fig1
@define
class LaiKaplanArray(Generic[T]):
    """
    Thin wrapper around numpy arrays to support using indexing like in the paper
    """

    min: float | int
    """Minimum index"""

    stride: float = field(validator=attrs.validators.in_((0.5, 1.0, 1)))
    """Size of stride"""

    data: NDArray[T]
    """Actual data array"""

    @property
    def max_allowed_lai_kaplan_index(self):
        return (self.data.size - 1) * self.stride + self.min

    def to_data_index(
        self, idx_lai_kaplan: int | float | None, is_slice_idx: bool = False
    ) -> int | None:
        if idx_lai_kaplan is None:
            return None

        if idx_lai_kaplan < self.min:
            msg = f"{idx_lai_kaplan=} is less than {self.min=}"
            raise IndexError(msg)

        idx_data_float = (idx_lai_kaplan - self.min) / self.stride
        if idx_data_float % 1.0:
            msg = f"{idx_lai_kaplan=} leads to {idx_data_float=}, which is not an int. {self=}"
            raise IndexError(msg)

        idx_data = int(idx_data_float)

        if is_slice_idx:
            max_idx = self.data.size
        else:
            max_idx = self.data.size - 1

        if idx_data > max_idx:
            msg = (
                f"{idx_lai_kaplan=} leads to {idx_data=}, "
                f"which is outside the bounds of `self.data` ({self.data.size=}). "
                f"{self.max_allowed_lai_kaplan_index=}, {self=}"
            )
            raise IndexError(msg)

        return idx_data

    def __getitem__(self, idx_lai_kaplan: int | float | slice) -> T | NDArray[T]:
        if isinstance(idx_lai_kaplan, slice):
            idx_data = slice(
                self.to_data_index(idx_lai_kaplan.start, is_slice_idx=True),
                self.to_data_index(idx_lai_kaplan.stop, is_slice_idx=True),
                self.to_data_index(idx_lai_kaplan.step, is_slice_idx=True),
            )

        else:
            idx_data = self.to_data_index(idx_lai_kaplan)

        return self.data[idx_data]


class MPIBoundaryHandling(StrEnum):
    CONSTANT = "constant"
    CUBIC_EXTRAPOLATION = "cubic_extrapolation"


def mean_preserving_interpolation(
    x_in: pint.UnitRegistry.Quantity,
    y_in: pint.UnitRegistry.Quantity,
    x_out: pint.UnitRegistry.Quantity,
    boundary_handling_left: MPIBoundaryHandling = MPIBoundaryHandling.CONSTANT,
    boundary_handling_right: MPIBoundaryHandling = MPIBoundaryHandling.CUBIC_EXTRAPOLATION,
) -> pint.UnitRegistry.Quantity:
    """
    Perform a mean-preserving interpolation

    Parameters
    ----------
    x_in
        x-values of the input

    y_in
        y-values of the input

    x_out
        x-values of the target x-grid

    boundary_handling_left
        Boundary handling to use on the left-hand end of the interval.

        If `MPIBoundaryHandling.CONSTANT`, we use a constant extrapolation
        of `y_in` to get a value one step to the left of the provided values.

        If `MPIBoundaryHandling.CUBIC_EXTRAPOLATION`, we fit a cubic spline
        to `x_in` and `y_in`, then use this to determine the value for `y_in`
        one step to the left of the provided values.

    boundary_handling_right
        Same as `boundary_handling_left`, but for the right-hand end of the interval.

    Returns
    -------
    y_out :
        Interpolated values for y, on the grid defined by `x_out`.
    """
    # Switch to raw arrays (could do this with pint too...)
    x_in_m = x_in.m
    y_in_m = y_in.m
    x_out_m = x_out.to(x_in.units).m

    x_in_m_diffs = x_in_m[1:] - x_in_m[:-1]
    x_in_m_diff = x_in_m_diffs[0]
    if not np.equal(x_in_m_diffs, x_in_m_diff).all():
        msg = "Non-uniform spacing in x"
        raise NotImplementedError(msg)

    if boundary_handling_left not in MPIBoundaryHandling:
        raise NotImplementedError(boundary_handling_left)

    if boundary_handling_right not in MPIBoundaryHandling:
        raise NotImplementedError(boundary_handling_right)

    x_extrap = np.hstack([x_in_m[0] - x_in_m_diff, x_in_m, x_in_m[-1] + x_in_m_diff])
    y_extrap = np.hstack([0, y_in_m, 0])

    boundary_handling = (boundary_handling_left, boundary_handling_right)
    if any(bh == MPIBoundaryHandling.CONSTANT for bh in boundary_handling):
        if boundary_handling_left == MPIBoundaryHandling.CONSTANT:
            y_extrap[0] = y_in_m[0]

        if boundary_handling_right == MPIBoundaryHandling.CONSTANT:
            y_extrap[-1] = y_in_m[-1]

    if any(bh == MPIBoundaryHandling.CUBIC_EXTRAPOLATION for bh in boundary_handling):
        cubic_interpolator = scipy.interpolate.interp1d(
            x_in_m, y_in_m, kind="cubic", fill_value="extrapolate"
        )

        if boundary_handling_left == MPIBoundaryHandling.CUBIC_EXTRAPOLATION:
            y_extrap[0] = cubic_interpolator(x_extrap[0])

        if boundary_handling_right == MPIBoundaryHandling.CUBIC_EXTRAPOLATION:
            y_extrap[-1] = cubic_interpolator(x_extrap[-1])

    # TODO: move into a notebook
    # import matplotlib.pyplot as plt
    #
    # plt.plot(x_in_m, y_in_m)
    # plt.scatter(x0, y0)
    # plt.scatter(xn_plus_one, yn_plus_one)
    # x_fine = np.arange(x0, xn_plus_one, x_in_m_diffs[0] / 20)
    # plt.show()

    # Look at Lai and Kaplan (https://journals.ametsoc.org/view/journals/atot/39/4/JTECH-D-21-0154.1.xml)
    # The boundary values are what they (confusingly) call x_i.
    # Whereas they don't have the concept of x in their paper.
    # We abstract around this, but it does make mapping from this code to their paper trickier.
    # TODO: update so the function takes in boundaries
    interval_boundaries = np.hstack(
        [x_in_m - x_in_m_diff / 2, x_in_m[-1] + x_in_m_diff / 2]
    )

    # Area under the curve in each interval
    A = (interval_boundaries[1:] - interval_boundaries[:-1]) * y_in_m

    control_points_wall = (y_extrap[:-1] + y_extrap[1:]) / 2

    # Cubic Hermite functions
    hermite_cubic = [
        [
            Polynomial((1, 0, -3, 2)),
            Polynomial((0, 0, 3, -2)),
        ],
        [
            Polynomial((0, 1, -2, 1)),
            Polynomial((0, 0, -1, 1)),
        ],
    ]

    # Quartic Hermite functions
    hermite_quartic = [
        [
            hermite_cubic[0][0].integ(),
            hermite_cubic[0][1].integ(),
        ],
        [
            hermite_cubic[1][0].integ(),
            hermite_cubic[1][1].integ(),
        ],
    ]

    a = np.array(
        [
            -0.5 * hermite_quartic[1][0](1),
            (
                hermite_quartic[0][0](1)
                + hermite_quartic[0][1](1)
                + 0.5 * hermite_quartic[1][0](1)
                - 0.5 * hermite_quartic[1][1](1)
            ),
            0.5 * hermite_quartic[1][1](1),
        ]
    )
    beta = np.array(
        [
            hermite_quartic[0][0](1)
            - 0.5 * hermite_quartic[1][0](1)
            - 0.5 * hermite_quartic[1][1](1),
            hermite_quartic[0][1](1)
            + 0.5 * hermite_quartic[1][0](1)
            + 0.5 * hermite_quartic[1][1](1),
        ]
    )

    # a-matrix
    A_mat = np.zeros((x_in_m.size, x_in_m.size))
    rows, cols = np.diag_indices_from(A_mat)
    A_mat[rows[1:], cols[:-1]] = a[0]
    A_mat[rows, cols] = a[1]
    A_mat[rows[:-1], cols[1:]] = a[2]

    # b-vector
    b = np.zeros_like(y_in_m)
    b[0] = (
        2 * A[0]
        - beta[0] * control_points_wall[0]
        - beta[1] * control_points_wall[1]
        - a[0] * y_extrap[0]
    )
    b[1:-1] = (
        2 * A[1:-1]
        - beta[0] * control_points_wall[1:-2]
        - beta[1] * control_points_wall[2:-1]
    )
    b[-1] = (
        2 * A[-1]
        - beta[0] * control_points_wall[-2]
        - beta[1] * control_points_wall[-1]
        - a[2] * y_extrap[-1]
    )

    control_points_mid = np.linalg.solve(A_mat, b)
    control_points = np.empty(
        control_points_wall.size + control_points_mid.size,
        dtype=control_points_wall.dtype,
    )
    control_points[0::2] = control_points_wall
    control_points[1::2] = control_points_mid

    # Now that we have all the control points, we can do the gradients
    gradients_at_control_points = control_points[2:] - control_points[:-2]

    x_in_half_intervals = np.empty(
        2 * interval_boundaries.size - 1, dtype=interval_boundaries.dtype
    )
    x_in_half_intervals[0::2] = interval_boundaries
    x_in_half_intervals[1::2] = (interval_boundaries[1:] + interval_boundaries[:-1]) / 2

    # Delta is half the interval width
    delta = x_in_m_diff / 2.0

    def get_filling_function(interval_idx: int) -> Callable[[float], float]:
        def filling_function(u: float) -> float:
            # TODO: change this to be use partial instead of higher-order function
            return (
                control_points[interval_idx] * hermite_cubic[0][0](u)
                + delta
                * gradients_at_control_points[interval_idx]
                * hermite_cubic[1][0](u)
                + control_points[interval_idx + 1] * hermite_cubic[0][1](u)
                + delta
                * gradients_at_control_points[interval_idx + 1]
                * hermite_cubic[1][1](u)
            )

        return filling_function

    interval_idx = 0
    filling_function = get_filling_function(interval_idx)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.step(x_in_m, y_in_m)
    for i in range(x_in_half_intervals.size - 1):
        x_l = x_in_half_intervals[i]
        x_u = x_in_half_intervals[i + 1]
        x_fine = np.linspace(x_l, x_u, 100)
        breakpoint()
        # idx = i + 1
        # filling_function = get_filling_function(i + 1)
        #
        # np.testing.assert_allclose(filling_function(0), control_points[idx])
        #
        # ax.plot(x_fine, filling_function(x_fine), label=i)
        # breakpoint()

    plt.show()
    breakpoint()

    for x_out_i in x_out_m:
        if x_out_i > x_in_half_intervals[interval_idx + 1]:
            interval_idx += 1
            filling_function = get_filling_function(interval_idx)

        elif x_out_i > x_in_half_intervals[interval_idx + 1]:
            raise NotImplementedError

        u = (x_out_i - x_in_half_intervals[interval_idx]) / delta
        # TODO: consider whether to simplify this.
        # If you're doing constant spacing, it can be less complex...
        breakpoint()

    # if weights is None:
    #     weights = np.ones_like(x)
    #
    # resolution_increase = int(x.size / X.size)
    #
    # degrees_freedom = int(np.ceil(degrees_freedom_scalar * Y.size))
    #
    # knots_prev = np.repeat(x[0], degree)
    # knots_post = np.repeat(x[-1], degree)
    # knots_internal = np.quantile(x, np.linspace(0, 1, degrees_freedom - degree + 1))
    # knots = np.hstack([knots_prev, knots_internal, knots_post])
    #
    # alpha_len = knots.size - degree
    #
    # B = np.column_stack(
    #     [
    #         np.ones(x.size),
    #         scipy.interpolate.BSpline.design_matrix(x, t=knots, k=degree).toarray(),
    #     ]
    # )
    #
    # if alpha_len != B.shape[1]:
    #     raise AssertionError
    #
    # BM = np.zeros((X.size, B.shape[1]))
    # for i in range(X.size):
    #     start_idx = i * resolution_increase
    #     stop_idx = (i + 1) * resolution_increase
    #     BM[i, :] = np.average(B[start_idx:stop_idx, :], axis=0, weights=weights[start_idx:stop_idx])
    #
    # BD = np.diff(B, axis=0)
    #
    # c = np.hstack([np.ones(BD.shape[0]), np.zeros(2 * alpha_len)])
    #
    # A_eq = np.column_stack([np.zeros((X.size, BD.shape[0])), BM, -BM])
    # b_eq = Y
    # A_ub = np.row_stack(
    #     [
    #         np.column_stack([-np.eye(BD.shape[0]), BD, -BD]),
    #         np.column_stack([-np.eye(BD.shape[0]), -BD, BD]),
    #     ]
    # )
    # b_ub = np.zeros(2 * BD.shape[0])
    #
    # res = scipy.optimize.linprog(
    #     c,
    #     A_eq=A_eq,
    #     b_eq=b_eq,
    #     A_ub=A_ub,
    #     b_ub=b_ub,
    #     method="highs",
    #     bounds=(0, None),
    #     # options=dict(maxiter=int(maxiter)),
    # )
    # if not res.success:
    #     raise AssertionError(res.message)
    #
    # alpha = res.x[-2 * alpha_len : -alpha_len] - res.x[-alpha_len:]
    # intercept = alpha[0]
    # coefficients = alpha[1:]
    #
    # return coefficients, intercept, knots, degree


def interpolate_time_slice_parallel_helper(
    inp: tuple[T, xr.DataArray],
    degrees_freedom_scalar: float = 1.75,
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

    degrees_freedom_scalar
        Degrees of freedom to use when calculating the interpolating spline.

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
    pint.set_application_registry(pint_xarray.accessors.default_registry)  # type: ignore

    return time, interpolate_lat_15_degree_to_half_degree(
        da.pint.quantify(), degrees_freedom_scalar=degrees_freedom_scalar
    )
