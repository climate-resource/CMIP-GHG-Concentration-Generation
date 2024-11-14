"""
Mean-preserving interpolation algorithms
"""

from __future__ import annotations

from enum import StrEnum
from functools import partial
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

    def to_data_index(self, idx_lai_kaplan: int | float | None, is_slice_idx: bool = False) -> int | None:
        if idx_lai_kaplan is None:
            return None

        if idx_lai_kaplan < self.lai_kaplan_idx_min:
            msg = f"{idx_lai_kaplan=} is less than {self.lai_kaplan_idx_min=}"
            raise IndexError(msg)

        idx_data_float = (idx_lai_kaplan - self.lai_kaplan_idx_min) / self.lai_kaplan_stride
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

    def __setitem__(self, idx_lai_kaplan: int | float | slice, val: T | npt.NDArray[T]) -> None:
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
    control_points_x_d[0 : control_points_x_d.size - 2 : 2] = x_bounds_in_m - (x_bounds_m_diff / 2)
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

        if boundary_handling_left == MPIBoundaryHandling.CUBIC_EXTRAPOLATION:
            cubic_interpolator = scipy.interpolate.interp1d(
                x_bounds_mid_points[:4],
                y_in_m[:4],
                kind="cubic",
                fill_value="extrapolate",
            )
            y_extrap_d[0] = cubic_interpolator(control_points_x_d[0])

        if boundary_handling_right == MPIBoundaryHandling.CUBIC_EXTRAPOLATION:
            cubic_interpolator = scipy.interpolate.interp1d(
                x_bounds_mid_points[-4:],
                y_in_m[-4:],
                kind="cubic",
                fill_value="extrapolate",
            )
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
        x_bounds_out=x_bounds_out.to(x_bounds_in.u).m,
    )

    y_out = type(y_in)(y_out_raw, y_in.units)

    return y_out


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

HERMITE_QUARTICS: tuple[tuple[Polynomial, Polynomial], tuple[Polynomial, Polynomial]] = tuple(
    tuple(hc.integ() for hc in hcs_row) for hcs_row in HERMITE_CUBICS
)
"""
Hermite quartic polynomials

Allows for the same notation as
[Lai and Kaplan, J. Atmos. Oceanic Technol. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1).
"""


def lai_kaplan_f(  # noqa: PLR0913
    x: T,
    x_i: T,
    s_i: T,
    s_i_plus_half: T,
    m_i: T,
    m_i_plus_half: T,
    delta: T,
) -> T:
    u = (x - x_i) / delta
    res = (
        s_i * HERMITE_CUBICS[0][0](u)
        + 2 * delta * m_i * HERMITE_CUBICS[1][0](u)
        + s_i_plus_half * HERMITE_CUBICS[0][1](u)
        + 2 * delta * m_i_plus_half * HERMITE_CUBICS[1][1](u)
    )

    return res


def mean_preserving_interpolation_lai_kaplan(  # noqa: PLR0913
    lai_kaplan_n: int,
    x: LaiKaplanArray[npt.NDArray[np.float64]],
    y: LaiKaplanArray[npt.NDArray[T]],
    y_extrap: LaiKaplanArray[npt.NDArray[T]],
    control_points_x: LaiKaplanArray[npt.NDArray[np.float64]],
    x_bounds_out: npt.NDArray[np.float64],
) -> npt.NDArray[T]:
    x_steps = x.data[1:] - x.data[:-1]
    x_step = x_steps[0]
    if not np.equal(x_steps, x_step).all():
        msg = "Non-uniform spacing in x"
        raise NotImplementedError(msg)

    delta = x_step / 2.0

    a_d = np.array(
        [
            -2 * delta * HERMITE_QUARTICS[1][0](1),
            (
                HERMITE_QUARTICS[0][0](1)
                + HERMITE_QUARTICS[0][1](1)
                + 2 * delta * HERMITE_QUARTICS[1][0](1)
                - 2 * delta * HERMITE_QUARTICS[1][1](1)
            ),
            2 * delta * HERMITE_QUARTICS[1][1](1),
        ]
    )
    a = LaiKaplanArray(
        lai_kaplan_idx_min=1,
        lai_kaplan_stride=1,
        data=a_d,
    )

    # Area under the curve in each interval
    A_d = (x.data[1:] - x.data[:-1]) * y[:]
    A = LaiKaplanArray(lai_kaplan_idx_min=1, lai_kaplan_stride=1, data=A_d)

    # beta array
    beta_d = np.array(
        [
            (
                HERMITE_QUARTICS[0][0](1)
                - 2 * delta * HERMITE_QUARTICS[1][0](1)
                - 2 * delta * HERMITE_QUARTICS[1][1](1)
            ),
            (
                HERMITE_QUARTICS[0][1](1)
                + 2 * delta * HERMITE_QUARTICS[1][0](1)
                + 2 * delta * HERMITE_QUARTICS[1][1](1)
            ),
        ]
    )
    beta = LaiKaplanArray(1, 1, beta_d)

    # a-matrix
    # (Not indexed in the paper, hence not done with Lai Kaplan indexing)
    A_mat = np.zeros((lai_kaplan_n, lai_kaplan_n))
    rows, cols = np.diag_indices_from(A_mat)
    A_mat[rows[1:], cols[:-1]] = a[1]
    A_mat[rows, cols] = a[2]
    A_mat[rows[:-1], cols[1:]] = a[3]

    # b-vector
    # Give the user the ability to change this
    control_points_wall_y_d = (
        y_extrap[0 : lai_kaplan_n + 1 : 1] + y_extrap[1 : lai_kaplan_n + 1 + 1 : 1]
    ) / 2
    first_increase = np.argmax(~np.isclose(control_points_wall_y_d[:-1], control_points_wall_y_d[1:]))
    if first_increase > 0:
        control_points_wall_y_d[first_increase + 1] = control_points_wall_y_d[0]

    control_points_wall_y = LaiKaplanArray(
        lai_kaplan_idx_min=1,
        lai_kaplan_stride=1,
        data=control_points_wall_y_d,
    )

    b = LaiKaplanArray(
        lai_kaplan_idx_min=1,
        lai_kaplan_stride=1,
        data=np.zeros_like(y.data),
    )
    b[1] = (
        2 * A[1]
        - beta[1] * control_points_wall_y[1]
        - beta[2] * control_points_wall_y[2]
        - a[1] * y_extrap[0]
    )
    middle_slice = slice(2, lai_kaplan_n)
    middle_slice_plus_one = slice(3, lai_kaplan_n + 1)
    b[middle_slice] = (
        2 * A[middle_slice]
        - beta[1] * control_points_wall_y[middle_slice]
        - beta[2] * control_points_wall_y[middle_slice_plus_one]
    )
    b[lai_kaplan_n] = (
        2 * A[lai_kaplan_n]
        - beta[1] * control_points_wall_y[lai_kaplan_n]
        - beta[2] * control_points_wall_y[lai_kaplan_n + 1]
        - a[3] * y_extrap[lai_kaplan_n + 1]
    )

    control_points_mid_y_d = np.linalg.solve(A_mat, b.data)

    control_points_y = LaiKaplanArray(
        lai_kaplan_idx_min=1 / 2,
        lai_kaplan_stride=1 / 2,
        data=np.zeros_like(control_points_x.data),
    )

    control_points_y[1 / 2] = y_extrap[0]
    control_points_y[1 : lai_kaplan_n + 1 + 1 : 1] = control_points_wall_y[:]
    control_points_y[3 / 2 : lai_kaplan_n + 1 / 2 + 1 / 2 : 1] = control_points_mid_y_d
    control_points_y[lai_kaplan_n + 3 / 2] = y_extrap[lai_kaplan_n + 1]

    # Now that we have all the control points, we can do the gradients
    gradients_at_control_points = LaiKaplanArray(
        lai_kaplan_idx_min=1,
        lai_kaplan_stride=1 / 2,
        data=np.zeros(2 * lai_kaplan_n + 1),
    )
    gradients_at_control_points[:] = (
        control_points_y[3 / 2 : lai_kaplan_n + 1 + 1] - control_points_y[1 / 2 : lai_kaplan_n + 1]
    )

    for i in range(x.data.size - 1):
        interval_idx = i + 1

        ff_l = partial(
            lai_kaplan_f,
            s_i=control_points_y[interval_idx],
            s_i_plus_half=control_points_y[interval_idx + 1 / 2],
            m_i=gradients_at_control_points[interval_idx],
            m_i_plus_half=gradients_at_control_points[interval_idx + 1 / 2],
            delta=delta,
            x_i=x.data[i],
        )
        ff_u = partial(
            lai_kaplan_f,
            s_i=control_points_y[interval_idx + 1 / 2],
            s_i_plus_half=control_points_y[interval_idx + 1],
            m_i=gradients_at_control_points[interval_idx + 1 / 2],
            m_i_plus_half=gradients_at_control_points[interval_idx + 1],
            delta=delta,
            x_i=x.data[i] + delta,
        )
        int_l = scipy.integrate.quad(ff_l, x.data[i], x.data[i] + delta)[0]
        int_u = scipy.integrate.quad(ff_u, x.data[i] + delta, x.data[i] + 2 * delta)[0]
        # np.testing.assert_allclose(int_l + int_u, A[interval_idx], atol=1e-10)
        # np.testing.assert_allclose(int_l + int_u, delta * y[interval_idx], atol=1e-10)

    interval_idx = 1
    x_i = x[interval_idx]
    filling_function = partial(
        lai_kaplan_f,
        s_i=control_points_y[interval_idx],
        s_i_plus_half=control_points_y[interval_idx + 1 / 2],
        m_i=gradients_at_control_points[interval_idx],
        m_i_plus_half=gradients_at_control_points[interval_idx + 1 / 2],
        delta=delta,
        x_i=x_i,
    )
    np.testing.assert_allclose(control_points_y[interval_idx], filling_function(x_i))
    np.testing.assert_allclose(control_points_y[interval_idx + 1 / 2], filling_function(x_i + delta))

    # TODO: move to notebook
    plot = True
    # plot = False
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.step(x.data, [*y[:], y[:][-1]], linewidth=4, where="post", zorder=1)
        x_fine = np.linspace(x_i, x_i + delta, 100)

        ax.plot(x_fine, filling_function(x_fine), label=interval_idx)
        ax.scatter(control_points_x[:], control_points_y[:], zorder=3, label="CPs")

    if not np.all(x_bounds_out[:-1] <= x_bounds_out[1:]):
        msg = f"x_bounds_out must be sorted for this to work {x_bounds_out=}"
        raise AssertionError(msg)

    y_out = np.zeros(x_bounds_out.size - 1)
    for i in range(y_out.size):
        if x_bounds_out[i] >= control_points_x[interval_idx + 1 / 2]:
            interval_idx += 1 / 2
            in_middle_of_bound = interval_idx % 1.0 == 0.5  # noqa: PLR2004
            if in_middle_of_bound:
                x_i = x_i + delta

            else:
                x_i = x[interval_idx]

            filling_function = partial(
                lai_kaplan_f,
                s_i=control_points_y[interval_idx],
                s_i_plus_half=control_points_y[interval_idx + 1 / 2],
                m_i=gradients_at_control_points[interval_idx],
                m_i_plus_half=gradients_at_control_points[interval_idx + 1 / 2],
                delta=delta,
                x_i=x_i,
            )

            np.testing.assert_allclose(control_points_y[interval_idx], filling_function(x_i))
            np.testing.assert_allclose(control_points_y[interval_idx + 1 / 2], filling_function(x_i + delta))

            if plot:
                x_fine = np.linspace(x_i, x_i + delta, 100)
                integral = scipy.integrate.quad(filling_function, x_fine[0], x_fine[-1])[0]
                ax.plot(
                    x_fine,
                    filling_function(x_fine),
                    label=f"{interval_idx}: {integral=:.2f}",
                )

        integration_res = scipy.integrate.quad(filling_function, x_bounds_out[i], x_bounds_out[i + 1])
        y_out[i] = integration_res[0] / (x_bounds_out[i + 1] - x_bounds_out[i])

    def group_average(inarr, n):
        cumulative = np.cumsum(inarr)

        res = (
            np.hstack(
                [
                    cumulative[n - 1],
                    cumulative[2 * n - 1 :: n] - cumulative[n - 1 : -n : n],
                ]
            )
            / n
        )

        return res

    # TODO: figure out how to infer spacing rather than hard-coding
    res_increase = 12
    # TODO: should vary by gas, this should be the pre-industrial value
    # Might have to think about only making this apply over a certain time period too
    min_val = 0.0

    below_min = y_out < min_val
    below_min_in_group = group_average(below_min, 12)
    below_min_vals = below_min_in_group > 0
    needs_refinement = ~np.isclose(y.data, group_average(y_out, res_increase))
    # TODO: put in a check that nothing needs refinement and remove the refinement logic below

    intervals_to_polish = set([*np.where(needs_refinement)[0], *np.where(below_min_vals)[0]])

    # TODO: blend with the other Rymes-Meyers stuff below
    adjust_mat = np.zeros((res_increase, res_increase))
    rows, cols = np.diag_indices_from(adjust_mat)
    adjust_mat[rows[1:], cols[:-1]] = 1 / 3
    adjust_mat[rows, cols] = 1 / 3
    adjust_mat[rows[:-1], cols[1:]] = 1 / 3

    for polish_interval in intervals_to_polish:
        lai_kaplan_interval = polish_interval + 1
        interval_slice = slice(polish_interval * res_increase, (polish_interval + 1) * res_increase)
        interval_vals = y_out[interval_slice]

        interval_vals_lt_min = interval_vals < min_val
        if interval_vals_lt_min.any():
            y_h = y[lai_kaplan_interval]
            if (
                y_h == min_val
                and (control_points_y[lai_kaplan_interval] == min_val)
                and (control_points_y[lai_kaplan_interval + 1] == min_val)
            ):
                # Better if we could fast exit here, but ok
                interval_new = min_val

            if lai_kaplan_interval == 1:
                turning_point = True
            elif lai_kaplan_interval == lai_kaplan_n:
                turning_point = True
            else:
                y_prev = y[lai_kaplan_interval - 1]
                y_next = y[lai_kaplan_interval + 1]
                # Don't need to check positive turning point as we're only applying a lower limit
                turning_point = (y_h <= y_prev) and (y_h <= y_next)

            if not turning_point:
                msg = "How did this happen?"
                raise AssertionError(msg)

            tmp = np.zeros(interval_vals.size + 2)
            tmp[0] = y_out[interval_slice.start - 1]
            tmp[1:-1] = min_val
            tmp[-1] = y_out[interval_slice.stop]

            max_it = int(1e3)
            for i in range(max_it):
                interval_new = tmp[1:-1]
                if not (interval_new < min_val).any():
                    # Worth checkiing for convergence
                    interval_new_mean = np.mean(interval_new)

                    if np.isclose(interval_new_mean, y_h):
                        # All values above min and close to target, all done
                        break

                    if interval_new_mean < y_h:
                        # Too low still, let the iterations continue
                        pass

                    else:
                        # We've overshot
                        overshoot = interval_new_mean - y_h
                        distance_to_min = interval_new - min_val
                        adjustable = overshoot <= distance_to_min
                        if adjustable.all():
                            # Adjust all the values down to match the mean
                            interval_new -= overshoot
                            interval_new_mean = np.mean(interval_new)
                            if np.isclose(interval_new_mean, y_h) and np.all(interval_new_mean >= min_val):
                                break
                            else:
                                msg = "Should match the mean with no values below min by construction now"
                                raise AssertionError(msg)
                        else:
                            # Use the Rymes-Myers updating algorithm
                            # Don't run the Lai-Kaplan updating algorithm
                            correction = (
                                np.sum(interval_new - y_h)
                                / np.sum(interval_new - min_val)
                                * (interval_new - min_val)
                            )
                            tmp[1:-1] = interval_new - correction
                            tmp[np.where(tmp < min_val)] = min_val
                            continue

                tmp[1:-1] = adjust_mat @ tmp[1:-1]
                tmp[1] += tmp[0] / 3
                tmp[-2] += tmp[-1] / 3

            else:
                msg = f"Ran out of iterations for {polish_interval=}, check for kinks"
                raise AssertionError(msg)

            y_out[polish_interval * res_increase : (polish_interval + 1) * res_increase] = interval_new

        else:
            raise NotImplementedError

    if not np.isclose(group_average(y_out, res_increase), y.data).all():
        msg = "Should be close now"
        raise AssertionError(msg)

    if plot:
        ax.step(x_bounds_out, [*y_out, y_out[-1]], linewidth=2, where="post")
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        ax.grid()
        plt.show()

    return y_out
