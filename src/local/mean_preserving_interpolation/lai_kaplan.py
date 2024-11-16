"""
Lai-Kaplan mean-preserving interpolator

See [Lai and Kaplan, J. Atmos. Oceanic Technol. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1)
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Generic, TypeVar

import attrs.validators
import numpy as np
import numpy.typing as npt
import pint
from attrs import define, field
from numpy.polynomial import Polynomial

from local.mean_preserving_interpolation.boundary_handling import (
    BoundaryHandling,
)
from local.optional_dependencies import get_optional_dependency

T = TypeVar("T")


@define
class LaiKaplanArray(Generic[T]):
    """
    Thin wrapper around numpy arrays to support indexing like in the paper.

    This is sort of like writing a Python array that supports Fortran-style indexing,
    but trying to translate the paper with raw python indexes was too confusing,
    so we wrote this instead.
    """

    lai_kaplan_idx_min: float | int
    """Minimum index"""

    lai_kaplan_stride: float = field(validator=attrs.validators.in_((0.5, 1.0, 1)))
    """Size of stride"""

    data: npt.NDArray[T]
    """Actual data array"""

    @property
    def max_allowed_lai_kaplan_index(self):
        """
        The maximum allowed Lai-Kaplan style index for `self.data`

        Returns
        -------
        :
            The index.
        """
        return (self.data.size - 1) * self.lai_kaplan_stride + self.lai_kaplan_idx_min

    def to_data_index(self, idx_lai_kaplan: int | float | None, is_slice_idx: bool = False) -> int | None:
        """
        Convert a Lai-Kaplan index to the equivalent index for `self.data`

        Parameters
        ----------
        idx_lai_kaplan
            Lai-Kaplan index to translate

        is_slice_idx
            Whether this index is a slice index.

            This is important to ensure we give sensible errors
            about whether the index is too big for `self.data` or not.

        Returns
        -------
        :
            The index for `self.data`.
        """
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
        """
        Translate a Lai-Kaplan step into the equivalent step for `self.data`

        Parameters
        ----------
        step_lai_kaplan
            Lai-Kaplan step size

        Returns
        -------
        :
            `self.data` step size
        """
        if step_lai_kaplan is None:
            return None

        step_data_float = step_lai_kaplan / self.lai_kaplan_stride
        if step_data_float % 1.0:
            msg = f"{step_lai_kaplan=} leads to {step_data_float=}, which is not an int. {self=}"
            raise IndexError(msg)

        step_data = int(step_data_float)

        return step_data

    def __getitem__(self, idx_lai_kaplan: int | float | slice) -> T | npt.NDArray[T]:
        """
        Get an item from `self.data` using standard Python indexing

        The trick here is that we can use indexing like in the Lai-Kaplan paper
        and get the correct part of the underlying data array back.
        """
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
        """
        Set an item (or slice) in `self.data` using standard Python indexing

        The trick here is that we can use indexing like in the Lai-Kaplan paper
        and set the correct part of the underlying data array.
        """
        if isinstance(idx_lai_kaplan, slice):
            idx_data = slice(
                self.to_data_index(idx_lai_kaplan.start, is_slice_idx=True),
                self.to_data_index(idx_lai_kaplan.stop, is_slice_idx=True),
                self.to_data_step(idx_lai_kaplan.step),
            )

        else:
            idx_data = self.to_data_index(idx_lai_kaplan)

        self.data[idx_data] = val


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

Allows for the same notation as the paper.
"""

HERMITE_QUARTICS: tuple[tuple[Polynomial, Polynomial], tuple[Polynomial, Polynomial]] = tuple(
    tuple(hc.integ() for hc in hcs_row) for hcs_row in HERMITE_CUBICS
)
"""
Hermite quartic polynomials

Allows for the same notation as the paper.
"""


@define
class LaiKaplanF:
    """
    Lai-Kaplan interpolating function
    """

    x_i: pint.UnitRegistry.Quantity
    """Start of the interval over which this function applies"""

    delta: pint.UnitRegistry.Quantity
    """Size of the domain over which this function applies"""

    s_i: pint.UnitRegistry.Quantity
    """Value at the left-hand edge of the domain (`x = x_i`)"""

    s_i_plus_half: pint.UnitRegistry.Quantity
    """Value at the right-hand edge of the domain (`x = x_i + delta`)"""

    m_i: pint.UnitRegistry.Quantity
    """Gradient at the left-hand edge of the domain (`x = x_i`)"""

    m_i_plus_half: pint.UnitRegistry.Quantity
    """Gradient at the right-hand edge of the domain (`x = x_i + delta`)"""

    def calculate(
        self,
        x: pint.UnitRegistry.Quantity,
        check_domain: bool = True,
    ) -> pint.UnitRegistry.Quantity:
        """
        Calculate Lai-Kaplan interpolating function value

        Parameters
        ----------
        x
            Value for which we want to calculate the value of the function

        check_domain
            Whether to check that is in the supported domain before calculating.

        Returns
        -------
        :
            Function value at `x`, given the other parameters
        """
        if check_domain:
            if (x < self.x_i) or (x > self.x_i + self.delta):
                msg = f"x is outside the supported domain. {x=} {self.x_i=} {self.x_i + self.delta=}"
                raise ValueError(msg)

        u = (x - self.x_i) / self.delta

        return self.calculate_u(u, check_domain=False)

    def calculate_unitless(
        self,
        x: float,
        check_domain: bool = True,
    ) -> float:
        """
        Calculate Lai-Kaplan interpolating function value

        Do the calculation without units.
        This is helpful for integrating the function with scipy.

        Parameters
        ----------
        x
            Value for which we want to calculate the value of the function

        check_domain
            Whether to check that is in the supported domain before calculating.

        Returns
        -------
        :
            Function value at `x`, given the other parameters
        """
        if check_domain:
            if (x < self.x_i.m) or (x > self.x_i.m + self.delta.m):
                msg = f"x is outside the supported domain. {x=} {self.x_i=} {self.x_i + self.delta=}"
                raise ValueError(msg)

        u = (x - self.x_i.m) / self.delta.m

        res = (
            self.s_i.m * HERMITE_CUBICS[0][0](u)
            + self.delta.m * self.m_i.m * HERMITE_CUBICS[1][0](u)
            + self.s_i_plus_half.m * HERMITE_CUBICS[0][1](u)
            + self.delta.m * self.m_i_plus_half.m * HERMITE_CUBICS[1][1](u)
        )

        return res

    def calculate_u(
        self,
        u: float | pint.UnitRegistry.Quantity,
        check_domain: bool = True,
    ) -> pint.UnitRegistry.Quantity:
        """
        Calculate Lai-Kaplan interpolating function value

        Parameters
        ----------
        u
            Value for which we want to calculate the value of the function.

            This should have been normalised first i.e. this is in 'u-space', not 'x-space'.

        check_domain
            Whether to check that is in the supported domain before calculating.

        Returns
        -------
        :
            Function value at `u`, given the other parameters
        """
        if check_domain:
            if (u < 0) or (u > 1):
                msg = f"u is outside the supported domain. {u=}"
                raise ValueError(msg)

        res = (
            self.s_i * HERMITE_CUBICS[0][0](u)
            + self.delta * self.m_i * HERMITE_CUBICS[1][0](u)
            + self.s_i_plus_half * HERMITE_CUBICS[0][1](u)
            + self.delta * self.m_i_plus_half * HERMITE_CUBICS[1][1](u)
        )

        return res


def extrapolate_y_interval_values(
    x_in: pint.UnitRegistry.Quantity,
    y_in: pint.UnitRegistry.Quantity,
    x_out: pint.UnitRegistry.Quantity,
    left: BoundaryHandling = BoundaryHandling.CONSTANT,
    right: BoundaryHandling = BoundaryHandling.CUBIC_EXTRAPOLATION,
) -> pint.UnitRegistry.Quantity:
    """
    Extrapolate our y-interval values to get an extra value either side of the input domain

    Parameters
    ----------
    x_in
        x-values of the input array

    y_in
        y-values of the input array

    x_out
        x-values to extrapolate

        There should be two: the x-value to the left of `x_in`
        and the x-value to the right of `x_in`.

    left
        The extrapolation method to use for the left-hand value.

    right
        The extrapolation method to use for the right-hand value.

    Returns
    -------
    :
        The extrapolated values at `x_out`.
    """
    expected_out_size = 2

    if x_out.size != expected_out_size:
        raise NotImplementedError
    y_out = np.zeros(expected_out_size) * y_in.u

    if any(bh == BoundaryHandling.CUBIC_EXTRAPOLATION for bh in (left, right)):
        scipy_inter = get_optional_dependency("scipy.interpolate")

        cubic_interpolator = scipy_inter.interp1d(
            x_in.m,
            y_in.m,
            kind="cubic",
            fill_value="extrapolate",
        )
        if left == BoundaryHandling.CUBIC_EXTRAPOLATION:
            y_out[0] = cubic_interpolator(x_out[0].m) * y_in.u

        if right == BoundaryHandling.CUBIC_EXTRAPOLATION:
            y_out[-1] = cubic_interpolator(x_out[-1].m) * y_in.u

    else:
        if left == BoundaryHandling.CONSTANT:
            y_out[0] = y_in[0]
        else:
            raise NotImplementedError(left)

        if right == BoundaryHandling.CONSTANT:
            y_out[-1] = y_in[-1]
        else:
            raise NotImplementedError(right)

    return np.hstack(y_out)


@define
class LaiKaplanInterpolator:
    """
    Lai-Kaplan mean-preserving interpolator

    This splits each interval in half,
    then fits a mean-preserving cubic spline across each interval.
    The use of cubic splines means things can go a bit weird, but it is extremely fast.
    The option to specify minimum and maximum bounds for values
    allows you to limit some of the more extreme excursions.
    However, this boundary application is done using a Rymes-Meyers
    (see [`rymes_meyers`][local.mean_preserving_interpolation.rymes_meyers])
    style algorithm, so is much slower.

    See [Lai and Kaplan, J. Atm. Ocn. Tech. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1)
    """

    extrapolate_y_interval_values: Callable[
        [pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity],
        pint.UnitRegistry.Quantity,
    ] = partial(
        extrapolate_y_interval_values,
        left=BoundaryHandling.CONSTANT,
        right=BoundaryHandling.CUBIC_EXTRAPOLATION,
    )
    """
    Function that calculates the extrapolated y interval values from the input data

    This function is given the input y-values, plus the mid-point of each interval.
    """

    min_val: pint.UnitRegistry.Quantity | None = None
    """
    Minimum value that can appear in the solution
    """

    # min_it: int | None = None
    # """
    # Minimum number of iterations to perform
    #
    # If not provided, we use the length of the output data array.
    # """
    #
    # max_it: int = int(1e4)
    # """
    # Maximum number of iterations to perform
    # """

    atol: float = 1e-10
    """
    Absolute tolerance for deciding whether the output value means are close to the input means
    """

    rtol: float = 1e-7
    """
    Relative tolerance for deciding whether the output value means are close to the input means
    """

    progress_bar: bool = True
    """
    Whether to show a progress bar while filling the output array or not
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
        if not np.all(x_bounds_out[:-1] <= x_bounds_out[1:]):
            msg = f"x_bounds_out must be sorted for this to work {x_bounds_out=}"
            raise AssertionError(msg)

        x_steps = x_bounds_in[1:] - x_bounds_in[:-1]
        x_step = x_steps[0]
        if not np.equal(x_steps, x_step).all():
            msg = f"Non-uniform spacing in x {x_steps=}"
            raise NotImplementedError(msg)

        delta = x_step / 2.0
        intervals_internal_x = (x_bounds_in[1:] + x_bounds_in[:-1]) / 2.0
        walls_x = x_bounds_in
        external_intervals_x = np.hstack([x_bounds_in[0] - x_step / 2.0, x_bounds_in[-1] + x_step / 2.0])
        intervals_x = np.hstack(
            [
                external_intervals_x[0],
                intervals_internal_x,
                external_intervals_x[-1],
            ]
        )

        n_lai_kaplan = y_in.size

        # TODO: consider whether to make control points derivation injectable
        control_points_x_d = (
            np.zeros(
                2 * x_bounds_in.size + 1,
                # Has to be float so we can handle half steps even if input x array is integer
                dtype=np.float64,
            )
            * x_bounds_in.u
        )
        # Control points on the walls
        control_points_x_d[1::2] = walls_x
        # Control points in the intervals
        control_points_x_d[::2] = intervals_x

        control_points_x = LaiKaplanArray(
            lai_kaplan_idx_min=1 / 2,
            lai_kaplan_stride=1 / 2,
            data=control_points_x_d,
        )

        external_intervals_y = self.extrapolate_y_interval_values(
            x_in=control_points_x[1.5 : n_lai_kaplan + 1 : 1],
            y_in=y_in,
            x_out=np.hstack([control_points_x_d[0], control_points_x_d[-1]]),
        )
        intervals_y = LaiKaplanArray(
            lai_kaplan_idx_min=0.0,
            lai_kaplan_stride=1.0,
            data=np.hstack([external_intervals_y[0], y_in, external_intervals_y[-1]]),
        )

        # Control point values at the walls
        control_points_wall_y_d = (
            intervals_y[0 : n_lai_kaplan + 1 : 1] + intervals_y[1 : n_lai_kaplan + 1 + 1 : 1]
        ) / 2
        # If the values start out flat, keep them flat right until the end of the flat intervals.
        # TODO: allow the user to control this.
        first_increase = np.argmax(~np.isclose(control_points_wall_y_d[:-1], control_points_wall_y_d[1:]))
        if first_increase > 0:
            control_points_wall_y_d[first_increase + 1] = control_points_wall_y_d[0]

        control_points_wall_y = LaiKaplanArray(
            lai_kaplan_idx_min=1,
            lai_kaplan_stride=1,
            data=control_points_wall_y_d,
        )

        a_d = np.array(
            [
                -HERMITE_QUARTICS[1][0](1) / 2.0,
                (
                    HERMITE_QUARTICS[0][0](1)
                    + HERMITE_QUARTICS[0][1](1)
                    + HERMITE_QUARTICS[1][0](1) / 2.0
                    - HERMITE_QUARTICS[1][1](1) / 2.0
                ),
                HERMITE_QUARTICS[1][1](1) / 2.0,
            ]
        )
        a = LaiKaplanArray(
            lai_kaplan_idx_min=1,
            lai_kaplan_stride=1,
            data=a_d,
        )

        # A-matrix
        # (Not indexed in the paper, hence not done with Lai Kaplan indexing)
        A_mat = np.zeros((n_lai_kaplan, n_lai_kaplan))
        rows, cols = np.diag_indices_from(A_mat)
        A_mat[rows[1:], cols[:-1]] = a[1]
        A_mat[rows, cols] = a[2]
        A_mat[rows[:-1], cols[1:]] = a[3]

        # Area under the curve in each interval
        A_d = x_step * y_in
        A = LaiKaplanArray(lai_kaplan_idx_min=1, lai_kaplan_stride=1, data=A_d)

        # beta array
        beta_d = np.array(
            [
                (
                    HERMITE_QUARTICS[0][0](1)
                    - HERMITE_QUARTICS[1][0](1) / 2.0
                    - HERMITE_QUARTICS[1][1](1) / 2.0
                ),
                (
                    HERMITE_QUARTICS[0][1](1)
                    + HERMITE_QUARTICS[1][0](1) / 2.0
                    + HERMITE_QUARTICS[1][1](1) / 2.0
                ),
            ]
        )
        beta = LaiKaplanArray(1, 1, beta_d)

        b = LaiKaplanArray(
            lai_kaplan_idx_min=1,
            lai_kaplan_stride=1,
            data=np.zeros_like(y_in.data) * y_in.u,
        )
        b[1] = (
            A[1] / delta
            - beta[1] * control_points_wall_y[1]
            - beta[2] * control_points_wall_y[2]
            - a[1] * intervals_y[0]
        )
        middle_slice = slice(2, n_lai_kaplan)
        middle_slice_plus_one = slice(3, n_lai_kaplan + 1)
        b[middle_slice] = (
            A[middle_slice] / delta
            - beta[1] * control_points_wall_y[middle_slice]
            - beta[2] * control_points_wall_y[middle_slice_plus_one]
        )
        b[n_lai_kaplan] = (
            A[n_lai_kaplan] / delta
            - beta[1] * control_points_wall_y[n_lai_kaplan]
            - beta[2] * control_points_wall_y[n_lai_kaplan + 1]
            - a[3] * intervals_y[n_lai_kaplan + 1]
        )

        control_points_interval_y_d = np.linalg.solve(A_mat, b.data)

        control_points_y = LaiKaplanArray(
            lai_kaplan_idx_min=1 / 2,
            lai_kaplan_stride=1 / 2,
            data=np.zeros_like(control_points_x.data) * control_points_interval_y_d.u,
        )
        control_points_y[1 : n_lai_kaplan + 1 + 1 : 1] = control_points_wall_y[:]
        control_points_y[3 / 2 : n_lai_kaplan + 1 / 2 + 1 : 1] = control_points_interval_y_d
        # Pre-calculated external interval values
        control_points_y[1 / 2] = external_intervals_y[0]
        control_points_y[n_lai_kaplan + 3 / 2] = external_intervals_y[-1]

        # Now that we have all the control points, we can do the gradients
        gradients_at_control_points = LaiKaplanArray(
            lai_kaplan_idx_min=1,
            lai_kaplan_stride=1 / 2,
            data=np.zeros(2 * n_lai_kaplan + 1) * (control_points_y.data.u / delta.u),
        )
        gradients_at_control_points[:] = (
            control_points_y[3 / 2 : n_lai_kaplan + 1 + 1] - control_points_y[1 / 2 : n_lai_kaplan + 1]
        ) / (2 * delta)

        y_out = np.zeros(x_bounds_out.size - 1) * y_in.u
        # TODO: Can't see how to do this with vectors, maybe someone smarter can.
        iterh = range(y_out.size)
        if self.progress_bar:
            tqdman = get_optional_dependency("tqdm.autonotebook")
            iterh = tqdman.tqdm(iterh, desc="Calculating output values")

        scipy_integrate = get_optional_dependency("scipy.integrate")

        lai_kaplan_interval_idx = 1 / 2
        x_i = control_points_x[lai_kaplan_interval_idx] - 10 * delta
        for out_index in iterh:
            if x_bounds_out[out_index] >= x_i + delta:
                lai_kaplan_interval_idx += 1 / 2

                x_i = control_points_x[lai_kaplan_interval_idx]
                lai_kaplan_f = LaiKaplanF(
                    x_i=x_i,
                    delta=delta,
                    s_i=control_points_y[lai_kaplan_interval_idx],
                    s_i_plus_half=control_points_y[lai_kaplan_interval_idx + 1 / 2],
                    m_i=gradients_at_control_points[lai_kaplan_interval_idx],
                    m_i_plus_half=gradients_at_control_points[lai_kaplan_interval_idx + 1 / 2],
                )

                # Do a single calculation to make sure we have the units right
                res_x_i = lai_kaplan_f.calculate(x_i)
                pint.testing.assert_allclose(
                    res_x_i,
                    control_points_y[lai_kaplan_interval_idx],
                )

            integration_res = (
                scipy_integrate.quad(
                    partial(lai_kaplan_f.calculate_unitless, check_domain=False),
                    x_bounds_out[out_index].m,
                    x_bounds_out[out_index + 1].m,
                )
                * res_x_i.u
                * x_bounds_out.u
            )
            y_out[out_index] = integration_res[0] / (x_bounds_out[out_index + 1] - x_bounds_out[out_index])

        if self.min_val is not None and (y_out < self.min_val).any():
            raise NotImplementedError
            # below_min = y_out < self.min_val
            # below_min_in_group = get_group_integrals(
            #     integrand_x_bounds=x_bounds_out,
            #     integrand_y=below_min,
            #     group_bounds=x_bounds_in,
            # )
            # below_min_vals = below_min_in_group > 0
            breakpoint()

        return y_out
