"""
Mean-preserving interpolation algorithms
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Generic, TypeVar

import attrs.validators
import numpy as np
import numpy.typing as npt
import pint
import pint.testing
import scipy.interpolate  # type: ignore
import scipy.optimize  # type: ignore
from attrs import define, field
from numpy.polynomial import Polynomial

T = TypeVar("T")


@define
class LaiKaplanArray(Generic[T]):
    """
    Thin wrapper around numpy arrays to support using indexing like in the paper

    [Lai and Kaplan, J. Atmos. Oceanic Technol. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1)
    """

    lai_kaplan_idx_min: float | int
    """Minimum index"""

    lai_kaplan_stride: float = field(validator=attrs.validators.in_((0.5, 1.0, 1)))
    """Size of stride"""

    data: npt.NDArray[T]
    """Actual data array"""

    @property
    def max_allowed_lai_kaplan_index(self):
        return (self.data.size - 1) * self.lai_kaplan_stride + self.lai_kaplan_idx_min

    def to_data_index(
        self, idx_lai_kaplan: int | float | None, is_slice_idx: bool = False
    ) -> int | None:
        if idx_lai_kaplan is None:
            return None

        if idx_lai_kaplan < self.lai_kaplan_idx_min:
            msg = f"{idx_lai_kaplan=} is less than {self.lai_kaplan_idx_min=}"
            raise IndexError(msg)

        idx_data_float = (
            idx_lai_kaplan - self.lai_kaplan_idx_min
        ) / self.lai_kaplan_stride
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

    def to_data_step(self, step_lai_kaplan: int | float | None) -> int | None:
        if step_lai_kaplan is None:
            return None

        step_data_float = step_lai_kaplan / self.lai_kaplan_stride
        if step_data_float % 1.0:
            msg = f"{step_lai_kaplan=} leads to {step_data_float=}, which is not an int. {self=}"
            raise IndexError(msg)

        step_data = int(step_data_float)

        return step_data

    def __getitem__(self, idx_lai_kaplan: int | float | slice) -> T | npt.NDArray[T]:
        if isinstance(idx_lai_kaplan, slice):
            idx_data = slice(
                self.to_data_index(idx_lai_kaplan.start, is_slice_idx=True),
                self.to_data_index(idx_lai_kaplan.stop, is_slice_idx=True),
                self.to_data_step(idx_lai_kaplan.step),
            )

        else:
            idx_data = self.to_data_index(idx_lai_kaplan)

        return self.data[idx_data]

    def __setitem__(
        self, idx_lai_kaplan: int | float | slice, val: T | npt.NDArray[T]
    ) -> None:
        if isinstance(idx_lai_kaplan, slice):
            idx_data = slice(
                self.to_data_index(idx_lai_kaplan.start, is_slice_idx=True),
                self.to_data_index(idx_lai_kaplan.stop, is_slice_idx=True),
                self.to_data_step(idx_lai_kaplan.step),
            )

        else:
            idx_data = self.to_data_index(idx_lai_kaplan)

        self.data[idx_data] = val


class MPIBoundaryHandling(StrEnum):
    CONSTANT = "constant"
    CUBIC_EXTRAPOLATION = "cubic_extrapolation"


def mean_preserving_interpolation(
    x_bounds_in: pint.UnitRegistry.Quantity,
    y_in: pint.UnitRegistry.Quantity,
    x_bounds_out: pint.UnitRegistry.Quantity,
    boundary_handling_left: MPIBoundaryHandling = MPIBoundaryHandling.CONSTANT,
    boundary_handling_right: MPIBoundaryHandling = MPIBoundaryHandling.CUBIC_EXTRAPOLATION,
) -> pint.UnitRegistry.Quantity:
    """
    Perform a mean-preserving interpolation

    Uses the method from
    [Lai and Kaplan, J. Atmos. Oceanic Technol. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1),
    with a slight modification to handle output bounds, rather than points.

    Parameters
    ----------
    x_bounds_in
        The bounds of the x-values for each value in `y_in`

    y_in
        y-values of the input

    x_bounds_out
        The output/target bounds for the output

    boundary_handling_left
        Boundary handling to use on the left-hand end of the interval.

        If `MPIBoundaryHandling.CONSTANT`, we use a constant extrapolation
        of `y_in` to get a value one step to the left of the provided values.

        If `MPIBoundaryHandling.CUBIC_EXTRAPOLATION`, we fit a cubic spline
        to `x_bounds_in` and `y_in`, then use this to determine the value for `y_in`
        one step to the left of the provided values.

    boundary_handling_right
        Same as `boundary_handling_left`, but for the right-hand end of the interval.

    Returns
    -------
    y_out :
        Interpolated values for y, on the grid defined by `x_bounds_out`.
    """
    x_bounds_in_m = x_bounds_in.m
    y_in_m = y_in.m

    lai_kaplan_n = y_in.m.size

    x_bounds_in_m_diffs = x_bounds_in_m[1:] - x_bounds_in_m[:-1]
    x_bounds_m_diff = x_bounds_in_m_diffs[0]
    if not np.equal(x_bounds_in_m_diffs, x_bounds_m_diff).all():
        msg = "Non-uniform spacing in x"
        raise NotImplementedError(msg)

    control_points_x_d = np.zeros(
        2 * x_bounds_in_m.size + 1,
        # Has to be float so we can handle half steps even if input x array is integer
        dtype=np.float64,
    )
    control_points_x_d[1 : control_points_x_d.size - 1 : 2] = x_bounds_in_m
    control_points_x_d[0 : control_points_x_d.size - 2 : 2] = x_bounds_in_m - (
        x_bounds_m_diff / 2
    )
    control_points_x_d[-1] = x_bounds_in_m[-1] + x_bounds_m_diff / 2
    control_points_x = LaiKaplanArray(
        lai_kaplan_idx_min=1 / 2,
        lai_kaplan_stride=1 / 2,
        data=control_points_x_d,
    )

    if boundary_handling_left not in MPIBoundaryHandling:
        raise NotImplementedError(boundary_handling_left)

    if boundary_handling_right not in MPIBoundaryHandling:
        raise NotImplementedError(boundary_handling_right)

    y_extrap_d = np.hstack([0, y_in_m, 0])

    boundary_handling = (boundary_handling_left, boundary_handling_right)
    if any(bh == MPIBoundaryHandling.CONSTANT for bh in boundary_handling):
        if boundary_handling_left == MPIBoundaryHandling.CONSTANT:
            y_extrap_d[0] = y_in_m[0]

        if boundary_handling_right == MPIBoundaryHandling.CONSTANT:
            y_extrap_d[-1] = y_in_m[-1]

    if any(bh == MPIBoundaryHandling.CUBIC_EXTRAPOLATION for bh in boundary_handling):
        x_bounds_mid_points = (x_bounds_in_m[1:] + x_bounds_in_m[:-1]) / 2.0
        cubic_interpolator = scipy.interpolate.interp1d(
            x_bounds_mid_points, y_in_m, kind="cubic", fill_value="extrapolate"
        )

        if boundary_handling_left == MPIBoundaryHandling.CUBIC_EXTRAPOLATION:
            y_extrap_d[0] = cubic_interpolator(control_points_x_d[0])

        if boundary_handling_right == MPIBoundaryHandling.CUBIC_EXTRAPOLATION:
            y_extrap_d[-1] = cubic_interpolator(control_points_x_d[-1])

    # # TODO: move into a notebook
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots()
    # ax.step(x_bounds_in_m[:-1], y_in_m, linewidth=2)
    # ax.step(control_points_x.data[::2] - x_bounds_m_diff / 2, y_extrap_d, alpha=0.4)
    # ax.plot(control_points_x.data[::2] - x_bounds_m_diff, y_extrap_d, alpha=0.7)
    # ax.grid()
    # plt.show()

    y_extrap = LaiKaplanArray(
        lai_kaplan_idx_min=0,
        lai_kaplan_stride=1,
        data=y_extrap_d,
    )

    y_out_raw = mean_preserving_interpolation_lai_kaplan(
        lai_kaplan_n=lai_kaplan_n,
        x=LaiKaplanArray(lai_kaplan_idx_min=1, lai_kaplan_stride=1, data=x_bounds_in_m),
        y=LaiKaplanArray(lai_kaplan_idx_min=1, lai_kaplan_stride=1, data=y_in_m),
        y_extrap=y_extrap,
        control_points_x=control_points_x,
    )

    assert False

    # Switch to raw arrays (could do this with pint too...)
    x_bounds_in_m = x_in.m
    y_in_m = y_in.m
    x_out_m = x_out.to(x_in.units).m


HERMITE_CUBICS: tuple[tuple[Polynomial, Polynomial], tuple[Polynomial, Polynomial]] = (
    (
        Polynomial((1, 0, -3, 2), domain=[0, 1], window=[0, 1]),
        Polynomial((0, 0, 3, -2), domain=[0, 1], window=[0, 1]),
    ),
    (
        Polynomial((0, 1, -2, 1), domain=[0, 1], window=[0, 1]),
        Polynomial((0, 0, -1, 1), domain=[0, 1], window=[0, 1]),
    ),
)
"""
Hermite cubic polynomials

Allows for the same notation as
[Lai and Kaplan, J. Atmos. Oceanic Technol. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1).
"""

HERMITE_QUARTICS: tuple[
    tuple[Polynomial, Polynomial], tuple[Polynomial, Polynomial]
] = tuple(tuple(hc.integ() for hc in hcs_row) for hcs_row in HERMITE_CUBICS)
"""
Hermite quartic polynomials

Allows for the same notation as
[Lai and Kaplan, J. Atmos. Oceanic Technol. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1).
"""


def mean_preserving_interpolation_lai_kaplan(
    lai_kaplan_n: int,
    x: LaiKaplanArray[npt.NDArray[np.float64]],
    y: LaiKaplanArray[npt.NDArray[T]],
    y_extrap: LaiKaplanArray[npt.NDArray[T]],
    control_points_x: LaiKaplanArray[npt.NDArray[np.float64]],
) -> npt.NDArray[T]:
    a_d = np.array(
        [
            -0.5 * HERMITE_QUARTICS[1][0](1),
            (
                HERMITE_QUARTICS[0][0](1)
                + HERMITE_QUARTICS[0][1](1)
                + 0.5 * HERMITE_QUARTICS[1][0](1)
                - 0.5 * HERMITE_QUARTICS[1][1](1)
            ),
            0.5 * HERMITE_QUARTICS[1][1](1),
        ]
    )
    a = LaiKaplanArray(
        lai_kaplan_idx_min=1,
        lai_kaplan_stride=1,
        data=a_d,
    )

    # Area under the curve in each interval
    A_d = (x.data[1:] - x.data[:-1]) * y[:]
    A = LaiKaplanArray(1, 1, A_d)

    # beta array
    beta_d = np.array(
        [
            HERMITE_QUARTICS[0][0](1)
            - 0.5 * HERMITE_QUARTICS[1][0](1)
            - 0.5 * HERMITE_QUARTICS[1][1](1),
            HERMITE_QUARTICS[0][1](1)
            + 0.5 * HERMITE_QUARTICS[1][0](1)
            + 0.5 * HERMITE_QUARTICS[1][1](1),
        ]
    )
    beta = LaiKaplanArray(1, 1, beta_d)

    # a-matrix
    # (Not indexed in the paper, hence not done with Lai Kaplan indexing)
    A_mat = np.zeros((lai_kaplan_n, lai_kaplan_n))
    rows, cols = np.diag_indices_from(A_mat)
    A_mat[rows[1:], cols[:-1]] = a[0]
    A_mat[rows, cols] = a[1]
    A_mat[rows[:-1], cols[1:]] = a[2]

    # b-vector
    control_points_wall_y_d = (
        y_extrap[0 : lai_kaplan_n + 1 : 1] + y_extrap[1 : lai_kaplan_n + 1 + 1 : 1]
    ) / 2
    control_points_wall_y = LaiKaplanArray(
        lai_kaplan_idx_min=1,
        lai_kaplan_stride=1,
        data=control_points_wall_y_d,
    )
    breakpoint()

    b = np.zeros_like(y.data)
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

    breakpoint()

    control_points_mid = np.linalg.solve(A_mat, b)
    control_points = np.empty(
        control_points_wall_y.size + control_points_mid.size,
        dtype=control_points_wall_y.dtype,
    )
    control_points[0::2] = control_points_wall_y
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
                control_points[interval_idx] * HERMITE_CUBICS[0][0](u)
                + delta
                * gradients_at_control_points[interval_idx]
                * HERMITE_CUBICS[1][0](u)
                + control_points[interval_idx + 1] * HERMITE_CUBICS[0][1](u)
                + delta
                * gradients_at_control_points[interval_idx + 1]
                * HERMITE_CUBICS[1][1](u)
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
